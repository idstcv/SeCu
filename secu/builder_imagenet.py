# Copyright (c) Alibaba Group
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SeCu(nn.Module):
    """
    Build a SeCu model with multiple clustering heads
    """

    def __init__(self, base_encoder, K, dim=128, num_ins=1281167, t=0.05, alpha=90000, dual_lr=20,
                 lratio=0.4, constraint='size'):
        super(SeCu, self).__init__()
        self.K = K
        self.t = t
        self.num_head = len(self.K)
        self.alpha = alpha
        self.dual_lr = dual_lr
        self.lratio = lratio
        self.lbound = [lratio / curK for curK in self.K]
        self.cst = constraint
        # create the encoder with projection head
        self.encoder = base_encoder(num_classes=dim)
        dim_mlp = self.encoder.fc.weight.shape[1]

        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim), nn.BatchNorm1d(dim))
        # prediction head
        self.predictor = nn.Sequential(nn.Linear(dim, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                       nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim))

        # list for cluster assignments
        self.assign_labels = []
        self.counters = []
        for i in range(0, self.num_head):
            self.register_buffer("assign_labels_" + str(i), torch.ones(num_ins, dtype=torch.long))
            self.register_parameter("center_" + str(i), Parameter(F.normalize(torch.randn(dim, self.K[i]), dim=0)))
            self.register_buffer("counter_" + str(i), torch.zeros(self.K[i]))
        if self.cst == 'size':
            self.lduals = []
            for i in range(0, self.num_head):
                self.register_buffer("ldual_" + str(i), torch.zeros(self.K[i]))

    @torch.no_grad()
    def load_param(self):
        for i in range(0, self.num_head):
            self.assign_labels.append(getattr(self, "assign_labels_" + str(i)))
            self.counters.append(getattr(self, "counter_" + str(i)))
        if self.cst == 'size':
            for i in range(0, self.num_head):
                self.lduals.append(getattr(self, "ldual_" + str(i)))

    @torch.no_grad()
    def gen_label_size(self, obj_val, branch):
        return torch.argmin(obj_val - self.lduals[branch], dim=1).squeeze(-1)

    @torch.no_grad()
    def gen_label_entropy(self, obj_val, labels, branch, epoch):
        cur_count = self.counters[branch]
        if epoch == 0:
            base = torch.sum(cur_count)
            if base == 0:
                return torch.argmin(obj_val, dim=1).squeeze(-1)
            tmp_prob = cur_count / (base + 1.)
            if torch.min(tmp_prob) > 0:
                tmp_entropy = tmp_prob * torch.log(tmp_prob)
            else:
                tmp_entropy = torch.zeros(tmp_prob.shape, device=tmp_prob.device)
                tmp_entropy[tmp_prob>0] = tmp_prob[tmp_prob>0] * torch.log(tmp_prob[tmp_prob>0])
            tmp_prob_increase = tmp_prob + 1. / (base + 1.)
            tmp_entropy_increase = tmp_prob_increase * torch.log(tmp_prob_increase)
            entropy = tmp_entropy_increase - tmp_entropy
            return torch.argmin(obj_val + self.alpha * entropy.repeat(obj_val.shape[0], 1), dim=1).squeeze(-1)
        else:
            base = torch.sum(cur_count)
            tmp_prob = cur_count / base
            if torch.min(tmp_prob) > 0:
                tmp_entropy = tmp_prob * torch.log(tmp_prob)
            else:
                tmp_entropy = torch.zeros(tmp_prob.shape, device=tmp_prob.device)
                tmp_entropy[tmp_prob>0] = tmp_prob[tmp_prob>0] * torch.log(tmp_prob[tmp_prob>0])
            tmp_prob_increase = tmp_prob + 1. / base
            tmp_entropy_increase = tmp_prob_increase * torch.log(tmp_prob_increase)
            label_prob = tmp_prob[labels]
            label_entropy = label_prob * torch.log(label_prob)
            new_prob = label_prob - 1. / base
            if torch.min(new_prob) > 0:
                new_entropy = new_prob * torch.log(new_prob)
            else:
                new_entropy = torch.zeros(new_prob.shape, device=new_prob.device)
                new_entropy[new_prob>0] = new_prob[new_prob>0] * torch.log(new_prob[new_prob>0])
            entropy_reset = (new_entropy - label_entropy).reshape(-1, 1)
            entropy = entropy_reset - tmp_entropy + tmp_entropy_increase
            entropy.scatter_(1, labels.reshape(-1, 1), 0)
            return torch.argmin(obj_val + self.alpha * entropy, dim=1).squeeze(-1)

    @torch.no_grad()
    def update_label(self, targets, labels, branch):
        self.assign_labels[branch][targets] = labels

    @torch.no_grad()
    def get_label(self, target, branch):
        return self.assign_labels[branch][target]

    @torch.no_grad()
    def reset_count(self):
        for i in range(0, self.num_head):
            self.counters[i] *= 0


    @torch.no_grad()
    def update_count(self, labels, last_labels, branch, epoch):
        label_idx, label_count = torch.unique(labels, return_counts=True)
        if epoch == 0:
            self.counters[branch][label_idx] += label_count
        else:
            last_label_idx, last_label_count = torch.unique(last_labels, return_counts=True)
            self.counters[branch][label_idx] += label_count
            self.counters[branch][last_label_idx] -= last_label_count

    @torch.no_grad()
    def update_dual_mini_batch(self, labels, branch):
        label_idx, label_count = torch.unique(labels, return_counts=True)
        self.lduals[branch][label_idx] -= self.dual_lr / len(labels) * label_count
        self.lduals[branch] += self.dual_lr * self.lbound[branch]
        if self.lratio < 1:
            self.lduals[branch][self.lduals[branch] < 0] = 0
        self.counters[branch][label_idx] += label_count

    @torch.no_grad()
    def get_centers(self):
        centers = []
        for i in range(0, self.num_head):
            centers.append(F.normalize(getattr(self, "center_" + str(i)).clone().detach(), dim=0))
        return centers

    def forward(self, view1, view2, pre_centers, target, epoch, criterion, args):
        x1 = self.encoder(view1)
        x1_pred = F.normalize(self.predictor(x1), dim=1)
        x1_proj = F.normalize(x1, dim=1)
        x2 = self.encoder(view2)
        x2_pred = F.normalize(self.predictor(x2), dim=1)
        x2_proj = F.normalize(x2, dim=1)
        loss_proj_x = 0
        loss_pred_x = 0
        loss_proj_c = 0
        loss_pred_c = 0
        idx = torch.arange(len(target), device=target.device)
        targets = concat_all_gather(target)
        for i in range(0, self.num_head):
            cur_c = F.normalize(getattr(self, "center_" + str(i)), dim=0)
            proj_c1 = x1_proj.clone().detach() @ cur_c
            proj_c2 = x2_proj.clone().detach() @ cur_c
            pred_c1 = x1_pred.clone().detach() @ cur_c
            pred_c2 = x2_pred.clone().detach() @ cur_c
            # generate cluster assignments
            with torch.no_grad():
                pre_c = pre_centers[i]
                obj_val = -0.25 * (proj_c1 + proj_c2 + pred_c1 + pred_c2)
                if epoch == 0:
                    if self.cst == 'entropy':
                        label = self.gen_label_entropy(obj_val, None, i, epoch)
                        labels = concat_all_gather(label)
                        self.update_count(labels, None, i, epoch)
                    else:
                        label = self.gen_label_size(obj_val, i)
                        labels = concat_all_gather(label)
                        self.update_dual_mini_batch(labels, i)
                    self.update_label(targets, labels, i)
                    cur_label = self.get_label(target, i)
                else:
                    cur_label = self.get_label(target, i)
                    if self.cst == 'entropy':
                        label = self.gen_label_entropy(obj_val, cur_label, i, epoch)
                        labels = concat_all_gather(label)
                        self.update_count(labels, self.get_label(targets, i), i, epoch)
                    else:
                        label = self.gen_label_size(obj_val, i)
                        labels = concat_all_gather(label)
                        self.update_dual_mini_batch(labels, i)
                    self.update_label(targets, labels, i)

            # loss for cluster centers
            with torch.no_grad():
                logits_proj_c1 = proj_c1.clone().detach()
                logits_proj_c2 = proj_c2.clone().detach()
                logits_pred_c1 = pred_c1.clone().detach()
                logits_pred_c2 = pred_c2.clone().detach()
            logits_proj_c1[idx, label] = proj_c1[idx, label]
            logits_proj_c2[idx, label] = proj_c2[idx, label]
            logits_pred_c1[idx, label] = pred_c1[idx, label]
            logits_pred_c2[idx, label] = pred_c2[idx, label]
            loss_proj_c += criterion(logits_proj_c1 / self.t, label) + criterion(logits_proj_c2 / self.t, label)
            loss_pred_c += criterion(logits_pred_c1 / self.t, label) + criterion(logits_pred_c2 / self.t, label)

            # loss for representations
            proj_x1 = x1_proj @ pre_c / self.t
            proj_x2 = x2_proj @ pre_c / self.t
            pred_x1 = x1_pred @ pre_c / self.t
            pred_x2 = x2_pred @ pre_c / self.t
            with torch.no_grad():
                soft_label_view1 = (1.-args.secu_tau) * F.softmax(proj_x2, dim=1)
                soft_label_view2 = (1.-args.secu_tau) * F.softmax(proj_x1, dim=1)
                soft_label_view1[idx, cur_label] += args.secu_tau
                soft_label_view2[idx, cur_label] += args.secu_tau
            loss_proj_x -= (torch.mean(torch.sum(F.log_softmax(proj_x1, dim=1) * soft_label_view1, dim=1)) +
                torch.mean(torch.sum(F.log_softmax(proj_x2, dim=1) * soft_label_view2, dim=1)))
            loss_pred_x -= (torch.mean(torch.sum(F.log_softmax(pred_x1, dim=1) * soft_label_view1, dim=1)) +
                torch.mean(torch.sum(F.log_softmax(pred_x2, dim=1) * soft_label_view2, dim=1)))

        loss_x = (loss_proj_x + loss_pred_x) / (4. * self.num_head)
        loss_c = (loss_proj_c + loss_pred_c) / (4. * self.num_head)
        return loss_x, loss_c

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

