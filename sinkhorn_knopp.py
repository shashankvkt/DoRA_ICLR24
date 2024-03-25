import torch
import numpy as np


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 3:
                inp_tensor[ind[0], ind[1], ind[2]] = 0
            elif len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        # m = torch.max(inp_tensor)
        m = inp_tensor.max(-1).values.max(-1).values
        for ind in ind_inf:
            if len(ind) == 3:
                inp_tensor[ind[0], ind[1], ind[2]] = m[ind[0]]
            elif len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters_sk, epsilon_sk):
        super().__init__()
        self.num_iters = num_iters_sk
        self.epsilon = epsilon_sk

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = Q.sum(-1).sum(-1)
        Q /= sum_Q.unsqueeze(-1).unsqueeze(-1)

        B = Q.shape[2]
        K = Q.shape[1]
        
        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.transpose(-2, -1)

    @torch.no_grad()
    def forward(self, logits):
        # get assignments
        # import pdb; pdb.set_trace()
        qq = logits / self.epsilon
        M = qq.max(-1).values.max(-1).values
        qq -= M.unsqueeze(-1).unsqueeze(-1)
        qq = torch.exp(qq).transpose(-2, -1)
        return self.iterate(qq)
