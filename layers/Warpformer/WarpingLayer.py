import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from einops import rearrange, repeat

from utils.ExpConfigs import ExpConfigs

###############
### HELPERS ###
###############

def reset_parameters(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

class Constant(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes
        self.const = nn.parameter.Parameter(torch.Tensor(1, *output_sizes))

    # inp is an arbitrary tensor, whose values will be ignored;
    # output is self.const expanded over the first dimension of inp.
    # output.shape = (inp.shape[0], *output_sizes)
    def forward(self, inp):
        return self.const.expand(inp.shape[0], *((-1,)*len(self.output_sizes)))

    def reset_parameters(self):
        nn.init.uniform_(self.const, -1, 1) # U~[-1,1]

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
    def forward(self, inp):
        return inp.unsqueeze(self._dim)

class Square(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return inp*inp

class Abs(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.abs(inp)

class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.exp(inp)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.sin(inp)
    
#########################
### Localization net ###
#########################

    
class Scoring_Layer(nn.Module):
    def __init__(self, d_model, func_type='l2', active='relu'):
        super(Scoring_Layer, self).__init__()
        self.func_type = func_type

        assert func_type in ['l1','l2','l3','random','ones','pool'], print("func_type should in ['l1','l2','l3','random','ones','pool']")

        if active == 'relu':
            activation = nn.ReLU()
        elif active == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            activation = nn.Identity()

        if func_type == 'l1':
            self.reduce_d = nn.Linear(d_model, 1)
        elif func_type == 'l2':
            self.reduce_d = nn.Sequential(
                nn.Linear(d_model, d_model*2),
                activation,
                nn.Linear(d_model*2,1, bias=False))
        elif func_type == 'l3':
            self.reduce_d = nn.Sequential(
                nn.Linear(d_model, d_model*2),
                activation,
                nn.Linear(d_model*2,d_model*2),
                activation,
                nn.Linear(d_model*2,1, bias=False))
        elif func_type == 'pool':
            self.reduce_d = F.max_pool1d

    def forward(self, h, mask=None, mask_value=-1e30):
        # input: x: [b k l d], mask: [b k l]
        # output: score: [bk, l]
        b, k, l, d = h.shape
        # if h.dim() ==4:
        if self.func_type in ['l2', 'l1', "l3"]:
            h = self.reduce_d(h).squeeze(-1) # [b k l]
        elif self.func_type == 'pool':
            h = rearrange(h, 'b k l d -> (b k) l d')
            h = self.reduce_d(h, d).squeeze(-1)
        elif self.func_type == 'random':
            h = torch.rand((b*k, l), device=h.device)
        elif self.func_type == 'ones':
            h = torch.ones((b*k, l), device=h.device)

        if h.dim() == 3:
            h = rearrange(h, 'b k l -> (b k) l')
        return h


#########################
### WARPING FUNCTIONS ###
#########################

# backend can be any nn.Module that takes shape (batch_size, seq_len, input_size)
# and produces shape (batch_size, seq_len); the output of the backend is normalized
# and integrated.
class VanillaWarp(nn.Module):
    def __init__(self, backend, nonneg_trans='abs'):
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise ValueError("backend must be an instance of torch.nn.Module")
        self.backend = backend
        self.normintegral = NormalizedIntegral(nonneg_trans)

    # input_seq.shape = (b k l d)
    # output shape = (batch_size, seq_len)
    def forward(self, input_seq, mask):
        score = self.backend(input_seq, mask=mask) # [b k l d] -> [(b k) l]
        gamma = self.normintegral(score, mask) 
        return gamma

class NormalizedIntegral(nn.Module):
    # {abs, square, relu}      -> warping variance more robust to input variance
    # {exp, softplus, sigmoid} -> warping variance increases with input variance, strongest for exp
    def __init__(self, nonneg):
        super().__init__()
        # higher warping variance
        if nonneg == 'square':
            self.nonnegativity = Square()
        elif nonneg == 'relu':
            warnings.warn('ReLU non-negativity does not necessarily result in a strictly monotonic warping function gamma! In the worst case, gamma == 0 everywhere.', RuntimeWarning)
            self.nonnegativity = nn.ReLU()
        elif nonneg == 'exp':
            self.nonnegativity = Exp()
        # lower warping variance
        elif nonneg == 'abs':
            self.nonnegativity = Abs()
        elif nonneg == 'sigmoid':
            self.nonnegativity = nn.Sigmoid()
        elif nonneg == 'softplus':
            self.nonnegativity = nn.Softplus()
        else:
            raise ValueError("unknown non-negativity transformation, try: abs, square, exp, relu, softplus, sigmoid")

    # input_seq.shape = (batch_size, seq_len)
    # output shape    = (batch_size, seq_len)
    def forward(self, input_seq, mask):
        gamma = self.nonnegativity(input_seq)
        # transform sequences to alignment functions between 0 and 1
        # dgamma = torch.cat([torch.zeros((gamma.shape[0],1)).to(input_seq.device), gamma], dim=1) # fix entry to 0
        mask_mask = torch.ones(gamma.shape).to(input_seq.device)
        # mask = rearrange(mask, 'b k l -> (b k) l')
        mask_mask[:,0] = 0
        mask = mask * mask_mask
        dgamma = mask * gamma
        gamma = torch.cumsum(dgamma, dim=-1) * mask
        # gamma /= torch.max(gamma, dim=1)[0].unsqueeze(1)
        gamma_max = torch.max(gamma, dim=1)[0].unsqueeze(1)
        gamma_max[gamma_max==0] = 1
        gamma = gamma / gamma_max
        return gamma

##########################
### WARPING LAYER ###
##########################

class Almtx(nn.Module):
    def __init__(self, configs: ExpConfigs, K):
        super().__init__()
        self.S = K
        warpfunc = "l2"
        warpact = "relu"
        nonneg_trans = "sigmoid"
        loc_net = Scoring_Layer(configs.d_model, func_type=warpfunc, active=warpact)
        self.warp = VanillaWarp(loc_net, nonneg_trans=nonneg_trans)
        self.only_down = False
        # self.old_warp = opt.old_warp
        # self.mulmask = opt.mulmask
        # self.overlap = opt.overlap

    # gamma.shape    = (batch_size, warped_len), alignment functions mapping to [0,1]
    # original.shape = (batch_size, original_len, original_dim)
    # output.shape   = (batch_size, warped_len, original_dim)
    # almat.shape    = (batch_size, warped_len, original_len)

    def del_corner(self,mtx):
        mtx[mtx==False] = 0
        mtx[mtx==True] = 1
        mtx_u = torch.zeros_like(mtx)
        mtx_u[:,1:,:] = mtx[:,:-1,:]
        mtx_d = torch.zeros_like(mtx)
        mtx_d[:,:-1,:] = mtx[:,1:,:]
        mtx_l = torch.zeros_like(mtx)
        mtx_l[:,:,1:] = mtx[:,:,:-1]
        mtx_r = torch.zeros_like(mtx)
        mtx_r[:,:,:-1] = mtx[:,:,1:]

        new_mtx1 = mtx + mtx_u + mtx_r
        new_mtx2 = mtx + mtx_d + mtx_l

        new_mtx = torch.max(new_mtx2, new_mtx1)
        repeat_corner = torch.zeros_like(new_mtx)
        repeat_corner[:,1:,:] = new_mtx[:,:-1,:]
        repeat_corner = new_mtx + repeat_corner 
        mtx = mtx * ((new_mtx < 3) | (repeat_corner >5))
        return mtx

    def cal_new_bound(self,Rl, Rr, gamma):
        # cal new Rl
        B, S, L = gamma.shape
        mask = (Rr - gamma >= 0)
        vl, _ = torch.max(mask * gamma.detach(),-1)
        new_Rl = torch.min(vl,torch.arange(0, 1, 1/S).to(gamma.device)).unsqueeze(-1).expand(B,S,L)

        # cal new Rr
        mask = (gamma - Rl >= 0)
        mask[mask==False] = 10
        mask[mask==True] = 1
        vr, _ = torch.min(mask * gamma.detach(),-1)
        tmp_Rr = torch.max(vr,torch.arange(1/S, 1+1/S, 1/S).to(gamma.device)).unsqueeze(-1).expand(B,S,L)
        new_Rr = tmp_Rr.clone()
        new_Rr[:,-1] = tmp_Rr[:,-1] + 1e-4

        return new_Rl, new_Rr

    def get_boundary(self, gamma, Rl, Rr, mask):
        if self.only_down:
            new_Rl, new_Rr = Rl, Rr
        else:
            new_Rl, new_Rr = self.cal_new_bound(Rl, Rr, gamma)

        bound_mask = (gamma - new_Rl >= 0) & (new_Rr - gamma > 0) # [b s l]

        bound_mask = mask * bound_mask

        A = torch.threshold(gamma - new_Rl, 0, 0) + torch.threshold(new_Rr - gamma, 0, 0)

        return A, bound_mask



    def forward(self, input_seq, mask):
        mask = rearrange(mask,'b k l -> (b k) l')
        gamma = self.warp(input_seq,mask) # [BK,L]

        mask = repeat(mask,'b l -> b s l', s=self.S)
        _, L = gamma.shape
        gamma = repeat(gamma, 'b l -> b s l', s=self.S)

        Rl = torch.arange(0, 1, 1/self.S).unsqueeze(0).unsqueeze(-1).expand(1, self.S, L).to(gamma.device)
        Rr = torch.arange(1/self.S, 1+1/self.S, 1/self.S).unsqueeze(0).unsqueeze(-1).expand(1, self.S, L).to(gamma.device)

        A, bound_mask = self.get_boundary(gamma, Rl, Rr, mask)

        A_diag = A * bound_mask
        A_sum = A_diag.sum(dim=-1, keepdim=True)
        A_sum = torch.where(A_sum==0, torch.ones_like(A_sum), A_sum).to(A_sum.device)
        A_norm = A_diag / A_sum

        return bound_mask.float(), A_norm