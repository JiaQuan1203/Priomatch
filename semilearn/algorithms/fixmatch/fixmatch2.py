# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import os
import numpy as np
import torch.nn.functional as F
import torchvision


@ALGORITHMS.register('fixmatch')
class FixMatch(AlgorithmBase):

    """
        priomatch - from Jiaquan Wang
        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.center_mask_cutoff = torch.full((self.args.num_classes,), 1/self.args.num_classes).cuda()
        self.backward_mask_cutoff = torch.full((self.args.num_classes,), 1/self.args.num_classes).cuda()
        self.init_sample_logit = self.init_bank(self.args)
        self.sample_logit = self.init_sample_logit
        self.cluster_center_logit = torch.zeros(self.args.num_classes, self.args.num_classes)

    def init_bank(self, args):
        phi = 'clipvitL14'
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
        lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_{args.lb_imb_ratio}_seed{args.seed}_idx.npy')
        data_path = os.path.join(base_dir, 'data', args.dataset.lower())

        lb_idx = np.load(lb_dump_path)
        dset = getattr(torchvision.datasets, args.dataset.upper())
        dset = dset(data_path, train=True, download=True)
        data, targets = dset.data, dset.targets
        targets = np.array(targets)
        num_classes = args.num_classes
        one_hot = np.eye(num_classes)[targets]
        lb_targets = torch.tensor(one_hot[lb_idx])

        return lb_targets.cuda()

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            print(self.epochs)
            print(epoch)

            max_probs, label_of_max  = torch.max(self.sample_logit, dim=-1)
            mask = max_probs >= self.center_mask_cutoff[label_of_max]
            cluster_center_label = torch.arange(0, self.args.num_classes).cuda()
            relation_matrix = (torch.eq(cluster_center_label[:, None], label_of_max[mask][None, :])).int()
            row_sums = relation_matrix.sum(dim=1)
            weights = relation_matrix / row_sums.unsqueeze(1)
            self.cluster_center_logit = weights.float()@self.sample_logit[mask].float()
            print(self.cluster_center_logit)

            if epoch:
                max_per_row, _ = torch.max(self.cluster_center_logit, dim=1)
                mean_of_max_center_logit = torch.mean(max_per_row)
                min_of_max_center_logit = torch.min(max_per_row)
                print(mean_of_max_center_logit)
                print("----------------------")

                relation_matrix = (torch.eq(cluster_center_label[:, None], label_of_max[None, :])).int()
                row_sums = relation_matrix.sum(dim=1)
                weights = relation_matrix / row_sums.unsqueeze(1)
                mean_of_max_per_label = weights.float()@max_probs.float()
                mean_of_max = torch.mean(mean_of_max_per_label)
                max_of_mean_of_max_per_label = torch.max(mean_of_max_per_label)

                for category in range(self.args.num_classes):
                    self.center_mask_cutoff[category] = min_of_max_center_logit / max_per_row[category] * mean_of_max_center_logit
                    self.backward_mask_cutoff[category] = mean_of_max_per_label[category] / max_of_mean_of_max_per_label * mean_of_max

            print(f"center_mask_cutoff:{self.center_mask_cutoff}")
            print(f"backward_mask_cutoff:{self.backward_mask_cutoff}")

            self.sample_logit = self.init_sample_logit

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # compute mask
            mask1 = self.mask_production1(logits_x_ulb_w)

            # generate unlabeled targets using pseudo label hook
            pseudo_label1 = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            

            unsup_loss1 = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label1,
                                               'ce',
                                               mask=mask1)

            pseudo_label2 = self.cluster_label(logits_x_ulb_w)

            mask2 = self.mask_production2(logits_x_ulb_w)

            unsup_loss2 = self.consistency_loss(logits_x_ulb_w,
                                               pseudo_label2,
                                               'ce',
                                               mask=mask2)    
            # print(unsup_loss2)
            # print(unsup_loss1)
            #unsup_loss = unsup_loss1
            #unsup_loss = unsup_loss1 + unsup_loss2 * (self.epoch + 1) ** (-1/3)
            unsup_loss = unsup_loss1 + unsup_loss2   

            total_loss = sup_loss + self.lambda_u * unsup_loss
        
        if self.it%8 == 0:
            self.update_bank(probs_x_ulb_w)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask1.float().mean().item())
        return out_dict, log_dict
    

    def update_bank(self, probs_x_ulb_w):
        self.sample_logit = torch.cat([self.sample_logit, probs_x_ulb_w.detach()], dim=0)


    def mask_production1(self, logits):
        max_probs, label_of_max  = torch.max(logits, dim=-1)
        cutoffs = self.backward_mask_cutoff[label_of_max.long()]
        masks = max_probs >= cutoffs
        return masks.to(torch.int)

    def cluster_label(self, logits):
        _ , label_of_max  = torch.max(logits, dim=-1)
        labels = self.cluster_center_logit[label_of_max.long()]
        return labels
    
    def mask_production2(self, logits):
        max_probs, _ = torch.max(logits, dim=-1)
        masks = torch.sigmoid(10 * (max_probs - 0.5))
        return masks

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]