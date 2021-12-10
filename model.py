import torch
import torch.nn as nn
from math import pi

from config import config


class Model(nn.Module):
    def __init__(self, ent_num, rel_num):
        super(Model, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(config.gamma), requires_grad=False)
        self.ents = nn.Parameter(torch.arange(ent_num).unsqueeze(dim=0), requires_grad=False)
        self.ent_embd = nn.Embedding(ent_num, config.ent_dim)
        self.rel_embd = nn.Embedding(rel_num, config.rel_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embd.weight)
        nn.init.xavier_uniform_(self.rel_embd.weight)

    def get_pos_embd(self, pos_sample):
        h = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
        r = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
        t = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
        return h, r, t

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        raise NotImplementedError

class RotatE(Model):
    def __init__(self, ent_num, rel_num):
        super(RotatE, self).__init__(ent_num, rel_num)
        self.ent_embd_im = nn.Embedding(ent_num, config.ent_dim)
        nn.init.xavier_uniform_(self.ent_embd_im.weight)
        nn.init.uniform_(self.rel_embd.weight, a=-pi, b=pi)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h_re, h_im, r, t_re, t_im = self.get_pos_embd(pos_sample)
        rel_re = torch.cos(r)
        rel_im = torch.sin(r)
        if neg_sample is not None:
            neg_embd_re, neg_embd_im = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score_re = t_re * rel_re + t_im * rel_im
                score_im = t_im * rel_re - t_re * rel_im
            elif mode == "tail-batch":
                score_re = h_re * rel_re - h_im * rel_im
                score_im = h_re * rel_im + h_im * rel_re
            else:
                raise ValueError("mode %s not supported" % mode)
            score_re = score_re - neg_embd_re
            score_im = score_im - neg_embd_im
        else:
            score_re = h_re * rel_re - h_im * rel_im
            score_im = h_re * rel_im + h_im * rel_re
            score_re = score_re - t_re
            score_im = score_im - t_im
        score = torch.stack([score_re, score_im], dim=0).norm(dim=0)
        score = score.sum(dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h_re, r, t_re = super(RotatE, self).get_pos_embd(pos_sample)
        h_im = self.ent_embd_im(pos_sample[:, 0]).unsqueeze(1)
        t_im = self.ent_embd_im(pos_sample[:, 2]).unsqueeze(1)
        return h_re, h_im, r, t_re, t_im

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)

class RotatH(Model):
    def __init__(self, ent_num, rel_num):
        super(RotatH, self).__init__(ent_num, rel_num)
        self.ent_embd_im = nn.Embedding(ent_num, config.ent_dim)
        self.wr = nn.Embedding(rel_num, config.rel_dim)

        nn.init.xavier_uniform_(self.ent_embd_im.weight)
        nn.init.uniform_(self.rel_embd.weight, a=-pi, b=pi)
        nn.init.xavier_uniform_(self.wr.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h_re, h_im, r, t_re, t_im, w = self.get_pos_embd(pos_sample)
        rel_re = torch.cos(r)
        rel_im = torch.sin(r)
        if neg_sample is not None:
            neg_embd_re, neg_embd_im = self.get_neg_embd(neg_sample)

            wr_neg_re = (w * neg_embd_re).sum(dim=-1, keepdim=True)
            wr_neg_re_wr = wr_neg_re * w
            wr_neg_re_wr = neg_embd_re - wr_neg_re_wr
            wr_neg_im = (w * neg_embd_im).sum(dim=-1, keepdim=True)
            wr_neg_im_wr = wr_neg_im * w
            wr_neg_im_wr = neg_embd_im - wr_neg_im_wr

            if mode == "head-batch":
                wr_t_re = (w * t_re).sum(dim=-1, keepdim=True)
                wr_t_re_wr = wr_t_re * w
                wr_t_re_wr = t_re - wr_t_re_wr
                wr_t_im = (w * t_im).sum(dim=-1, keepdim=True)
                wr_t_im_wr = wr_t_im * w
                wr_t_im_wr = t_im - wr_t_im_wr

                score_re = wr_neg_re_wr * rel_re - wr_neg_im_wr * rel_im
                score_im = wr_neg_re_wr * rel_im + wr_neg_im_wr * rel_re
                score_re = score_re - wr_t_re_wr
                score_im = score_im - wr_t_im_wr
            elif mode == "tail-batch":
                wr_h_re = (w * h_re).sum(dim=-1, keepdim=True)
                wr_h_re_wr = wr_h_re * w
                wr_h_re_wr = h_re - wr_h_re_wr
                wr_h_im = (w * h_im).sum(dim=-1, keepdim=True)
                wr_h_im_wr = wr_h_im * w
                wr_h_im_wr = h_im - wr_h_im_wr

                score_re = wr_h_re_wr * rel_re - wr_h_im_wr * rel_im
                score_im = wr_h_re_wr * rel_im + wr_h_im_wr * rel_re
                score_re = score_re - wr_neg_re_wr
                score_im = score_im - wr_neg_im_wr
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            wr_h_re = (w * h_re).sum(dim=-1, keepdim=True)
            wr_h_re_wr = wr_h_re * w
            wr_h_re_wr = h_re - wr_h_re_wr
            wr_h_im = (w * h_im).sum(dim=-1, keepdim=True)
            wr_h_im_wr = wr_h_im * w
            wr_h_im_wr = h_im - wr_h_im_wr

            wr_t_re = (w * t_re).sum(dim=-1, keepdim=True)
            wr_t_re_wr = wr_t_re * w
            wr_t_re_wr = t_re - wr_t_re_wr
            wr_t_im = (w * t_im).sum(dim=-1, keepdim=True)
            wr_t_im_wr = wr_t_im * w
            wr_t_im_wr = t_im - wr_t_im_wr

            score_re = wr_h_re_wr * rel_re - wr_h_im_wr * rel_im
            score_im = wr_h_re_wr * rel_im + wr_h_im_wr * rel_re
            score_re = score_re - wr_t_re_wr
            score_im = score_im - wr_t_im_wr
        score = torch.stack([score_re, score_im], dim=0).norm(dim=0)
        score = score.sum(dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h_re, r, t_re = super(RotatH, self).get_pos_embd(pos_sample)
        h_im = self.ent_embd_im(pos_sample[:, 0]).unsqueeze(1)
        t_im = self.ent_embd_im(pos_sample[:, 2]).unsqueeze(1)
        w = self.wr(pos_sample[:, 1]).unsqueeze(dim=1)
        return h_re, h_im, r, t_re, t_im, w

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)
