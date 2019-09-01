from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from inspect import signature
import copy

from module.recurrent_units import RTNRnn
from module.weight_init import weight_init, manual_init


def model_config(func):
    def set_config(*args, **kwargs):
        init_sig = signature(func).bind(*args, **kwargs)
        init_sig.apply_defaults()
        params = init_sig.arguments
        del params['self']
        m_config = {}
        for key, value in params.items():
            m_config[key] = copy.deepcopy(value)
        func(*args, **kwargs)
        if hasattr(args[0], 'config'):
            args[0].config.update(m_config)
        else:
            args[0].config = m_config

    return set_config


def score_dist(t1, t2, p=2):
    return (torch.abs(t1 - t2) ** p).sum(dim=-1).neg()


def add_item_dim(users_emb, expected_dim=3):
    if len(users_emb.size()) < expected_dim:
        users_emb = users_emb.unsqueeze(-2)
    return users_emb


def weighted_mean(tensor, weight=None):
    if weight is not None:
        while len(weight.size()) < len(tensor.size()):
            weight = weight.unsqueeze(-1)
        tensor = tensor * weight
        tensor = tensor.sum() / (weight.sum() + 1e-8)
    else:
        tensor = tensor.mean()
    return tensor


def sbpr_loss(pos, neg):
    return -F.logsigmoid(pos - neg)


def l2(tensor):
    return tensor ** 2


def create_rtn_loss(n):
    def rtn_loss(predict, weight=None):
        losses = []
        weights = []
        for i in range(n):
            losses.append(sbpr_loss(predict[0::n + 1], predict[i + 1::n + 1]))
            if weight is not None:
                weights.append(weight[0::n + 1])
        losses = torch.cat(losses, dim=0)
        if weight is not None:
            weight = torch.cat(weights, dim=0)
        return weighted_mean(losses, weight)

    return rtn_loss


class ModEmbedding(nn.Embedding):

    def reset_parameters(self):
        if self.embedding_dim > 1:
            std = math.sqrt(0.1 / self.embedding_dim)
            nn.init.normal_(self.weight.data, 0, std)
        else:
            nn.init.constant_(self.weight.data, 0)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        if input is None:
            return self.weight.clone()
        return super().forward(input)


class UserShortTermPreferenceModeling(nn.Module):

    def __init__(self, n_items, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.i_emb = ModEmbedding(n_items, n_hidden, padding_idx=0)
        self.rnn = RTNRnn(n_hidden)

    def get_user_short_term_embedding(self, users_items, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(len(users_items))
        i_hist_inp = self.i_emb(users_items)
        u_short_emb, hidden = self.rnn(i_hist_inp, hidden)
        return u_short_emb, hidden

    def forward(self, users_items, pred_items, hidden=None):
        u_emb, hidden = self.get_user_short_term_embedding(users_items, hidden=hidden)
        return self.calculate_score(u_emb, pred_items), hidden

    def calculate_score(self, u_emb, pred_items):
        i_emb = self.i_emb(pred_items)
        u_emb = add_item_dim(u_emb)
        return score_dist(u_emb, i_emb)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(batch_size, self.n_hidden)

    def predict(self, users_items, pred_items, hidden=None):
        for t in range(0, users_items.size(1)):
            u_emb, hidden = self.get_user_short_term_embedding(users_items[:, t:t + 1], hidden=hidden)
        return self.calculate_score(u_emb, pred_items), hidden


class UserLongTermPreferenceModeling(nn.Module):

    def __init__(self, n_users, n_items, n_hidden):
        super().__init__()
        self.i_emb = ModEmbedding(n_items, n_hidden, padding_idx=0)
        self.u_emb = ModEmbedding(n_users, n_hidden)

    def forward(self, users, pred_items):
        return self.calculate_score(users, pred_items)

    def calculate_score(self, users, pred_items):
        i_emb = self.i_emb(pred_items)
        u_emb = self.u_emb(users)
        u_emb = add_item_dim(u_emb)
        return score_dist(u_emb, i_emb)

    def predict(self, users, pred_items):
        return self.calculate_score(users, pred_items)


class RTN(nn.Module):

    @model_config
    def __init__(self, n_users, n_items, n_hidden, l_reg=0.01, scale=5,
                 alpha=0.5, n_neg=9):
        super().__init__()
        self.lt = UserLongTermPreferenceModeling(n_users, n_items, n_hidden)
        self.st = UserShortTermPreferenceModeling(n_items, n_hidden)

        self.n_neg = n_neg
        self.n_hidden = n_hidden
        self.l_reg = l_reg
        self.alpha = alpha
        self.scale = scale
        self.objective = create_rtn_loss(n_neg)
        self.i_bias = ModEmbedding(n_items, 1, padding_idx=0)

    def combine_score(self, lt_score, st_score, pred_items):
        i_bias = self.i_bias(pred_items).squeeze(-1)
        score = lt_score * self.alpha + st_score * (1 - self.alpha)
        score = score * self.scale
        score = score + i_bias
        return score

    def forward(self, users, users_items, pred_items, weight=None, hidden=None):
        lt_score = self.lt(users, pred_items)
        st_score, hidden = self.st(users_items, pred_items, hidden=hidden)
        obj_loss = self.objective(self.combine_score(lt_score, st_score, pred_items), weight=weight)
        reg_loss = self.regularize(users, pred_items, weight=weight)
        return obj_loss, reg_loss, hidden

    def reset(self):
        self.apply(weight_init)
        self.apply(manual_init)

    def regularize(self, users, pred_items, weight=None):
        u_lt_reg = weighted_mean(l2(self.lt.u_emb(users).unsqueeze(1)), weight)
        i_lt_reg = weighted_mean(l2(self.lt.i_emb(pred_items)), weight)
        i_st_reg = weighted_mean(l2(self.st.i_emb(pred_items)), weight)
        i_bias_reg = weighted_mean(l2(self.i_bias(pred_items)), weight)
        return (u_lt_reg + i_lt_reg + i_st_reg + i_bias_reg) * self.l_reg

    def predict(self, users, users_items, candidates, hidden=None):
        lt_score = self.lt.predict(users, candidates)
        st_score, hidden = self.st.predict(users_items, candidates, hidden=hidden)
        return self.combine_score(lt_score, st_score, candidates), hidden

    def save(self, out_dir):
        state = {
            'config': self.config,
            'params': self.state_dict(),
        }
        torch.save(state, out_dir)

    @staticmethod
    def load(load_dir):
        state = torch.load(load_dir, map_location='cpu')
        model = RTN(**state['config'])
        model.load_state_dict(state['params'])
        return model
