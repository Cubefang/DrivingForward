import torch
from pytorch3d.transforms import matrix_to_euler_angles 

from .loss_util import compute_photometric_loss, compute_masked_loss
from .single_cam_loss import SingleCamLoss

from lpips import LPIPS

class MultiCamLoss(SingleCamLoss):
    """
    Class for multi-camera(spatio & temporal) loss calculation
    """
    def __init__(self, cfg, rank):
        super(MultiCamLoss, self).__init__(cfg, rank)

        self.lpips = LPIPS(net="vgg").cuda(rank)
    
    def compute_spatio_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None):
        """
        This function computes spatial loss.
        """        
        # self occlusion mask * overlap region mask
        spatio_mask = ref_mask * target_view[('overlap_mask', 0, scale)]
        loss_args = {
            'pred': target_view[('overlap', 0, scale)],
            'target': inputs['color',0, 0][:,cam, ...]         
        }        
        spatio_loss = compute_photometric_loss(**loss_args)
        
        target_view[('overlap_mask', 0, scale)] = spatio_mask         
        return compute_masked_loss(spatio_loss, spatio_mask) 

    def compute_spatio_tempo_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in self.frame_ids[1:]:

            pred_mask = ref_mask * target_view[('overlap_mask', frame_id, scale)]
            pred_mask = pred_mask * reproj_loss_mask 
            
            loss_args = {
                'pred': target_view[('overlap', frame_id, scale)],
                'target': inputs['color',0, 0][:,cam, ...]
            } 
            
            spatio_tempo_losses.append(compute_photometric_loss(**loss_args))
            spatio_tempo_masks.append(pred_mask)
        
        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 1)
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 1)    

        # for the loss, take minimum value between reprojection loss and identity loss(moving object)
        # for the mask, take maximum value between reprojection mask and overlap mask to apply losses on all the True values of masks.
        spatio_tempo_loss, _ = torch.min(spatio_tempo_losses, dim=1, keepdim=True)
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=1, keepdim=True)
     
        return compute_masked_loss(spatio_tempo_loss, spatio_tempo_mask) 
    
    def compute_pose_con_loss(self, inputs, outputs, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes pose consistency loss in "Full surround monodepth from multiple cameras"
        """        
        ref_output = outputs[('cam', 0)]
        ref_ext = inputs['extrinsics'][:, 0, ...]
        ref_ext_inv = inputs['extrinsics_inv'][:, 0, ...]
   
        cur_output = outputs[('cam', cam)]
        cur_ext = inputs['extrinsics'][:, cam, ...]
        cur_ext_inv = inputs['extrinsics_inv'][:, cam, ...] 
        
        trans_loss = 0.
        angle_loss = 0.
     
        for frame_id in self.frame_ids[1:]:
            ref_T = ref_output[('cam_T_cam', 0, frame_id)]
            cur_T = cur_output[('cam_T_cam', 0, frame_id)]    

            cur_T_aligned = ref_ext_inv@cur_ext@cur_T@cur_ext_inv@ref_ext

            ref_ang = matrix_to_euler_angles(ref_T[:,:3,:3], 'XYZ')
            cur_ang = matrix_to_euler_angles(cur_T_aligned[:,:3,:3], 'XYZ')

            ang_diff = torch.norm(ref_ang - cur_ang, p=2, dim=1).mean()
            t_diff = torch.norm(ref_T[:,:3,3] - cur_T_aligned[:,:3,3], p=2, dim=1).mean()

            trans_loss += t_diff
            angle_loss += ang_diff
        
        pose_loss = (trans_loss + 10 * angle_loss) / len(self.frame_ids[1:])
        return pose_loss

    def compute_gaussian_loss(self, inputs, target_view, cam=0, scale=0):
        """
        This function computes gaussian loss.
        """
        # self occlusion mask * overlap region mask
        if self.novel_view_mode == 'MF':
            pred = target_view[('gaussian_color', 0, scale)]
            target_view = inputs['color', 0, 0][:,cam, ...]        
            lpips_loss = self.lpips(pred, target_view, normalize=True).mean()
            l2_loss = ((pred - target_view)**2).mean()
            return 1 * l2_loss + 0.05 * lpips_loss
        elif self.novel_view_mode == 'SF':
            gaussian_loss = 0.0 
            for frame_id in self.frame_ids[1:]:
                pred = target_view[('gaussian_color', frame_id, scale)]
                gt = inputs['color', frame_id, 0][:,cam, ...]        
                lpips_loss = self.lpips(pred, gt, normalize=True).mean()
                l2_loss = ((pred - gt)**2).mean()
                gaussian_loss += 1 * l2_loss + 0.05 * lpips_loss
            return gaussian_loss / 2


    def forward(self, inputs, outputs, cam):        
        loss_dict = {}
        cam_loss = 0. # loss across the multi-scale
        target_view = outputs[('cam', cam)]
        for scale in self.scales:
            kargs = {
                'cam': cam,
                'scale': scale,
                'ref_mask': inputs['mask'][:,cam,...]
            }
                          
            reprojection_loss = 0.0
            smooth_loss = 0.0
            spatio_loss = 0.0
            spatio_tempo_loss = 0.0
            if getattr(self, 'enable_self_supervised_depth', True):
                reprojection_loss = self.compute_reproj_loss(inputs, target_view, **kargs)
                smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)
                spatio_loss = self.compute_spatio_loss(inputs, target_view, **kargs)
                kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
                spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs)

            if self.gaussian:
                gaussian_loss = self.compute_gaussian_loss(inputs, target_view, cam, scale)
            
            pose_loss = 0
                
            if getattr(self, 'enable_self_supervised_depth', True):
                cam_loss += reprojection_loss
                cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)            
                cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss
            if self.gaussian:
                cam_loss += self.gaussian_coeff * gaussian_loss                            
            cam_loss += self.pose_loss_coeff* pose_loss

            # DVGT教师监督
            dvgt_loss = None
            if getattr(self, 'use_dvgt_supervision', False) and (('teacher_depth', 0, scale) in target_view):
                pred_depth = target_view[('depth', 0, scale)]
                gt_depth = target_view[('teacher_depth', 0, scale)]
                mask = target_view.get(('teacher_conf', 0, scale), None)
                if mask is not None:
                    dvgt_loss = (torch.abs(pred_depth - gt_depth) * mask).sum() / (mask.sum() + 1e-8)
                else:
                    dvgt_loss = torch.mean(torch.abs(pred_depth - gt_depth))
                cam_loss += getattr(self, 'dvgt_depth_coeff', 1.0) * dvgt_loss
            
            ##########################
            # for logger
            ##########################
            if scale == 0:
                if getattr(self, 'enable_self_supervised_depth', True):
                    loss_dict['reproj_loss'] = reprojection_loss if isinstance(reprojection_loss, float) else reprojection_loss.item()
                    loss_dict['spatio_loss'] = spatio_loss if isinstance(spatio_loss, float) else spatio_loss.item()
                if self.gaussian:
                    loss_dict['gaussian_loss'] = gaussian_loss.item()
                if getattr(self, 'enable_self_supervised_depth', True):
                    loss_dict['spatio_tempo_loss'] = spatio_tempo_loss if isinstance(spatio_tempo_loss, float) else spatio_tempo_loss.item()
                    loss_dict['smooth'] = smooth_loss if isinstance(smooth_loss, float) else smooth_loss.item()
                if dvgt_loss is not None:
                    loss_dict['dvgt_depth'] = dvgt_loss.item()
                
                # log statistics
                self.get_logs(loss_dict, target_view, cam)                        
        
        cam_loss /= len(self.scales)
        return cam_loss, loss_dict