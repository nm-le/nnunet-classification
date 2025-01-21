import numpy as np
import torch
from torch import autocast, nn
from torch import distributed as dist
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from nnunetv2.utilities.collate_outputs import collate_outputs
from typing import List


class nnUNetTrainerMultiTask(nnUNetTrainer):
    """
    Custom nnUNet Trainer for multi-task learning: segmentation and classification.
    """
    
    def initialize(self):
        
        super().initialize()

    
    def train_step(self, batch: dict) -> dict:
        """
        Extends the training step to handle both segmentation and classification.
        """
        data = batch['data'].to(self.device, non_blocking=True)
        seg_target = batch['target']
        if isinstance(seg_target, list):
            seg_target = [t.to(self.device, non_blocking=True) for t in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)
        
        cls_labels = batch['class_labels'].to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            # Forward pass: get segmentation and classification outputs
            seg_outputs, cls_logits = self.network(data)
            
            # Compute segmentation loss
            seg_loss = self.loss(seg_outputs, seg_target)
            
            # print(cls_logits)
            # print(cls_labels)
            # Compute classification loss
            # class_weights = [0.7, 0.1, 0.2]
            class_weights = [0.5, 0.2, 0.3]
            class_weights = torch.FloatTensor(class_weights)

            criterion = nn.CrossEntropyLoss()
            cls_loss = criterion(cls_logits, cls_labels)
            # print(cls_loss)
            # import pdb; pdb.set_trace()
           
            # Combine losses
            total_loss = seg_loss + 0.3 * cls_loss
        
        # Backward pass and optimization
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': total_loss.detach().cpu().numpy()}
    
   

    def validation_step(self, batch: dict) -> dict:
        """
        Extends the validation step to handle both segmentation and classification.
        """
        data = batch['data']
        target = batch['target']
        cls_labels = batch['class_labels']
    
        data = data.to(self.device, non_blocking=True)
        cls_labels = cls_labels.to(self.device, non_blocking=True)
    
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
    
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set.
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True):
            # Forward pass: get segmentation and classification outputs
            seg_outputs, cls_logits = self.network(data)
            del data
            
            # Compute segmentation loss
            seg_loss = self.loss(seg_outputs, target)
            
            # Compute classification loss
            # cls_loss = nn.functional.cross_entropy(cls_logits, cls_labels)
            # class_weights = [0.5, 0.2, 0.3]
            class_weights = [0.5, 0.2, 0.3]
            class_weights = torch.FloatTensor(class_weights)

            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            cls_loss = criterion(cls_logits, cls_labels)
            # print('validation step')
            # print(f"class logits: {cls_logits}")
            # print(f"class labels: {cls_labels}")
            
            # Combine losses
            total_loss = seg_loss + 0.3 * cls_loss
            
            # Compute classification accuracy
            _, predicted = torch.max(cls_logits, 1)
            correct = (predicted == cls_labels).sum().item()
            total_cls = cls_labels.size(0)
            cls_accuracy = correct / total_cls if total_cls > 0 else 0.0
            print(cls_accuracy)
            
            # Segmentation Metrics
            # Only consider the highest resolution output if deep supervision is enabled
            if self.enable_deep_supervision:
                seg_output = seg_outputs[0]
                target = target[0]
            else:
                seg_output = seg_outputs
            
            # Define axes for Dice computation: batch and spatial dimensions
            axes = [0] + list(range(2, seg_output.ndim))
            
            if self.label_manager.has_regions:
                # Binary segmentation: apply sigmoid and threshold
                predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
            else:
                # Multi-class segmentation: apply argmax over channel dimension
                output_seg = seg_output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(seg_output.shape, device=seg_output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg
    
            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target != self.label_manager.ignore_label).float()
                    # CAREFUL: Modify target after applying mask
                    target[target == self.label_manager.ignore_label] = 0
                else:
                    if target.dtype == torch.bool:
                        mask = ~target[:, -1:]
                    else:
                        mask = 1 - target[:, -1:]
                    # CAREFUL: Modify target after applying mask
                    target = target[:, :-1]
            else:
                mask = None
    
            # Compute TP, FP, FN
            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
    
            # Convert to NumPy for aggregation
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_manager.has_regions:
                # Exclude background from metrics
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
    
            return {
                'loss': total_loss.detach().cpu().numpy(),
                'cls_accuracy': cls_accuracy,
                'tp_hard': tp_hard,
                'fp_hard': fp_hard,
                'fn_hard': fn_hard,
            }
        
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        Aggregates validation metrics across batches.
        """

        outputs_collated = collate_outputs(val_outputs)
        cls_accuracy = np.mean(outputs_collated['cls_accuracy'])
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_cls_accuracy', cls_accuracy, self.current_epoch)
