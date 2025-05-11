# base on https://github.com/lucidrains/gotennet-pytorch.git and https://github.com/sarpaykent/GotenNet.git

from functools import partial
import inspect
import math
from typing import Union
from einops import reduce, repeat
from einops.layers.torch import Rearrange
import einx
from einx import get_at
from e3nn.o3 import spherical_harmonics
import torch
from torch.nn import Module, Sequential, ModuleList
from torch import Tensor, nn, einsum, cat
import torch.nn.functional as F
from e3nn.o3 import spherical_harmonics

# ein notation

# b - batch
# h - heads
# n - sequence
# m - sequence (neighbors)
# i, j - source and target sequence
# d - feature
# m - order of each degree
# l - degree
# c - number of each degree
# ===========================
# variables
# z - atom_symbol
# h - 0-degree
# X - high-degree
# n - number of atoms
# k - number of neighbors
# r - number of radial basis
# ===========================
# easy functions
zeros_initializer = partial(nn.init.constant_, val=0.0)
xavier_uniform = partial(nn.init.xavier_uniform_)
LayerNorm = partial(nn.LayerNorm, bias=False)
Linear = partial(nn.Linear)
LinearNoBias = partial(nn.Linear, bias=False)


# function
def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def str2init(init_str="xavier_uniform"):
    if init_str == "":
        # No-op
        return lambda x: x
    elif init_str == "zeros":
        return nn.init.zeros_
    elif init_str == "xavier_uniform":
        return nn.init.xavier_uniform_
    elif init_str == "orthogonal":
        return nn.init.orthogonal_
    elif init_str == "kaiming_uniform":
        return nn.init.kaiming_uniform_
    else:
        raise ValueError(f"Unknown initialization {init_str}")


def shifted_softplus(x: Tensor):
    return F.softplus(x) - math.log(2.0)


def str2act(act_str="relu"):
    if act_str == "relu":
        return nn.ReLU()
    elif act_str == "elu":
        return nn.ELU()
    elif act_str == "sigmoid":
        return nn.Sigmoid()
    elif act_str == "silu":
        return nn.SiLU()
    elif act_str == "mish":
        return nn.Mish()
    elif act_str == "swish":
        return nn.SiLU()
    elif act_str == "selu":
        return nn.SELU()
    elif act_str == "softplus":
        return shifted_softplus
    else:
        raise ValueError(f"Unknown activation function {act_str}")


def str2gate(gate_str="silu"):
    if gate_str == "silu":
        return nn.Sigmoid()
    elif gate_str == "sigmoid":
        return nn.SiLU()
    elif gate_str == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function {gate_str}")


class GaussianRBF(Module):
    def __init__(self, cutoff: float = 5.0, n_rbf: int = 16, start: float = 0.0, trainable: bool = True):
        super().__init__()
        self.n_rbf = n_rbf

        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(torch.abs(offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: Tensor):
        coeff = -0.5 / torch.pow(self.widths, 2)
        diff = inputs[..., None] - self.offsets
        y = torch.exp(coeff * torch.pow(diff, 2))
        return y


class CosineCutoff(Module):
    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(Module):

    def __init__(self, cutoff: float = 5.0, n_rbf: int = 16, trainable: bool = False):
        super(ExpNormalSmearing, self).__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.n_rbf)
        betas = torch.tensor([(2 / self.n_rbf * (1 - start_value)) ** -2] * self.n_rbf)
        return means, betas

    def forward(self, dist: Tensor):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class BesselBasis(Module):
    def __init__(self, cutoff: float = 5.0, n_rbf: int = 16, trainable: bool = False):
        super().__init__()
        self.n_rbf = n_rbf
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        if trainable:
            self.register_buffer("freqs", nn.Parameter(freqs))
            self.register_buffer("widths", nn.Parameter(torch.tensor(1.0)))
        else:
            self.register_buffer("freqs", freqs)
            self.register_buffer("widths", torch.tensor(1.0))

    def forward(self, inputs: Tensor):
        a = self.freqs[None, :]
        inputs = inputs[..., None]
        ax = inputs * a
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, self.norm1, inputs)
        y = sinax / norm
        return y


class RadialBasis(Module):
    def __init__(self, dim, radial_hidden_dim=64):
        super().__init__()

        hidden = radial_hidden_dim

        self.rp = Sequential(
            Rearrange("... -> ... 1"),
            Linear(1, hidden),
            nn.SiLU(),
            LayerNorm(hidden),
            Linear(hidden, hidden),
            nn.SiLU(),
            LayerNorm(hidden),
            Linear(hidden, dim),
        )

    def forward(self, x: Tensor):
        return self.rp(x)


def str2basis(basis_str="gaussian"):
    if basis_str == "gaussian":
        return GaussianRBF
    elif basis_str == "expnorm":
        return ExpNormalSmearing
    elif basis_str == "bessel":
        return BesselBasis
    elif basis_str == "mlp":
        return RadialBasis
    else:
        raise ValueError(f"Unknown basis function {basis_str}")


def str2norm(input_str, dim):
    if input_str == "":
        return nn.Identity()
    if input_str == "layer":
        return torch.nn.LayerNorm(dim)
    if input_str == "batch":
        return torch.nn.BatchNorm1d(dim)
    if input_str == "instance":
        return torch.nn.InstanceNorm1d(dim)


def create_graph(coors: Tensor, cutoff: float = 5.0, max_neighbors: int = 32, mask: Tensor = None):
    self_loop_mask = torch.eye(coors.shape[1], device=coors.device, dtype=torch.bool)  # n n
    rel_pos = einx.subtract("b i c, b j c -> b i j c", coors, coors)  # b n n 3
    rel_dist = rel_pos.norm(dim=-1)  # b n n
    mask = einx.multiply("b i, b j -> b i j", mask, mask)  # b n n
    rel_dist = einx.where("b i j, b i j, -> b i j", mask, rel_dist, 1e6)  # b n n
    rel_dist = einx.where("i j, b i j, -> b i j", ~self_loop_mask, rel_dist, 1e6)  # b n n 消除自环和padding
    max_neighbors = min(max_neighbors, rel_dist.shape[-1])
    neighbor_index = rel_dist.topk(max_neighbors, dim=-1, largest=False).indices  # b n k
    neighbor_dist = rel_dist.gather(-1, neighbor_index)
    neighbor_mask = neighbor_dist <= cutoff
    neighbor_vec = rel_pos.gather(-2, repeat(neighbor_index, "... -> ... c", c=3))
    return neighbor_vec, neighbor_dist, neighbor_index, neighbor_mask  # b n k 3, b n k, b n k, b n k


def vector_rejection(rep: list, rl_ij: list, lmax: int):
    vec_porj = [einsum("... c d, ... c -> ... c", rep[i], rl_ij[i]) for i in range(lmax)]
    vec = [einx.multiply("... c, ... c -> ... c", vec_porj[i], rl_ij[i]) for i in range(lmax)]
    return [einx.subtract("... c d, ... c -> ... c d", rep[i], vec[i]) for i in range(lmax)]


# class
class Dense(nn.Linear):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        activation: str = None,
        dropout: float = 0.0,
        weight_init=xavier_uniform,
        bias_init=zeros_initializer,
        norm: str = None,
        gain=None,
    ):
        # initialize linear layer y = xW^T + b
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super().__init__(in_dim, out_dim, bias)
        self.activation = str2act(activation) if activation is not None else nn.Identity()
        self.norm = str2norm(norm, out_dim) if norm is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs: Tensor):
        y = super(Dense, self).forward(inputs)
        y = self.norm(y)
        y = self.activation(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):

    def __init__(
        self,
        hidden_dims: list,
        bias: bool = True,
        activation: str = None,
        last_activation: str = None,
        weight_init=xavier_uniform,
        bias_init=zeros_initializer,
        norm: str = "",
    ):
        super().__init__()

        n_layers = len(hidden_dims)

        DenseMLP = partial(Dense, bias=bias, weight_init=weight_init, bias_init=bias_init)

        self.dense_layers = ModuleList(
            [DenseMLP(hidden_dims[i], hidden_dims[i + 1], activation=activation, norm=norm) for i in range(n_layers - 2)]
            + [DenseMLP(hidden_dims[-2], hidden_dims[-1], activation=last_activation)]
        )

        self.layers = Sequential(*self.dense_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dense_layers:
            m.reset_parameters()

    def forward(self, x: Tensor):
        return self.layers(x)


class NodeInit(Module):

    def __init__(
        self,
        hidden_dim,
        n_rbf,
        cutoff,
        max_z=14,
        is_embed=False,
        activation="silu",
        proj_ln="layer",
        weight_init=xavier_uniform,
        bias_init=zeros_initializer,
    ):
        super().__init__()
        self.embed = Linear(max_z, hidden_dim) if is_embed else nn.Embedding(max_z, hidden_dim)  # A_na

        self.W_ndp = Dense(n_rbf, hidden_dim, weight_init=weight_init, bias_init=bias_init)  # W_ndp

        self.W_nrd_nru = MLP(
            [2 * hidden_dim] + [hidden_dim, hidden_dim],
            activation=activation,
            norm=proj_ln,
            last_activation=None,
            weight_init=weight_init,
            bias_init=bias_init,
        )  # W_nrd_nru

        self.cutoff = CosineCutoff(cutoff)
        self.reset_parameters()

    def reset_parameters(self):
        self.embed.reset_parameters()
        self.W_ndp.reset_parameters()
        self.W_nrd_nru.reset_parameters()

    def forward(
        self,
        z: Tensor,
        h: Tensor,
        neighbor_index: Tensor,
        neighbor_dist: Tensor,  # r0_ij
        neighbor_rb: Tensor,  # varphi_r0_ij
        neighbor_mask: Tensor = None,
    ):
        """
        NodeInit
        Args:
            z (Tensor): [batch, nums_atoms, ] or [batch, nums_atoms, input_dim]
            h (Tensor): [batch, nums_atoms, hidden_dim]
            neighbor_index (Tensor): [batch, nums_atoms, max_neighbors]
            neighbor_dist (Tensor): [batch, nums_atoms, max_neighbors]
            phi_r0_ij (Tensor): [batch, nums_atoms, max_neighbors, n_rbf]
            neighbor_mask (Tensor, optional): [batch, nums_atoms, max_neighbors]. Defaults to None.
        """
        h_src = self.embed(z)  # b a d
        dist_cutoff = self.cutoff(neighbor_dist)  # b a k
        r0_ij_feat = self.W_ndp(neighbor_rb) * dist_cutoff.unsqueeze(-1)  # b a k d
        # message
        neighbor_feat = get_at("b [n] d, b i j -> b i j d", h_src, neighbor_index)  # h_src_j * r0_ij_feat
        neighbor_feat = einx.where("b i j, b i j d, -> b i j d", neighbor_mask, neighbor_feat, 0.0)  # b a k d
        m_i = einsum("b i j d, b i j d -> b i d", neighbor_feat, r0_ij_feat)  # b a d aggr
        return self.W_nrd_nru(torch.cat((h, m_i), dim=-1))  # b a d


class EdgeInit(Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        n_rbf: int = 16,
        activation: str = None,
        weight_init=xavier_uniform,
        bias_init=zeros_initializer,
    ):
        super().__init__()
        self.W_erp = Dense(n_rbf, hidden_dim, weight_init=weight_init, bias_init=bias_init)  # W_erp
        self.activation = str2act(activation) if activation is not None else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        self.W_erp.reset_parameters()

    def forward(self, h: Tensor, neighbor_index: Tensor, neighbor_rb: Tensor, neighbor_mask: Tensor):
        # b n d, b n k, b n k r, b n k
        hj = get_at("b [n] d, b i j -> b i j d", h, neighbor_index)  # b n k d
        h_ij = einx.add("b i d, b i j d -> b i j d", h, hj)  # b n k d
        r_ij = self.W_erp(neighbor_rb)  # b n k d
        return h_ij * self.activation(r_ij)  # b n k d


class TensorInit(Module):
    def __init__(self, lmax=2):
        super().__init__()
        self.lmax = lmax
        self.spherical_harmonics = spherical_harmonics

    def forward(self, hs: Tensor, neighbor_vec: Tensor):
        rl_ij = [
            self.spherical_harmonics(l, neighbor_vec, normalize=True, normalization="norm") for l in range(1, self.lmax + 1)
        ]  # 1, ..., l
        # init high degree
        X = [torch.zeros((*hs[:-1], 2 * l + 1, hs[-1]), device=neighbor_vec.device) for l in range(1, self.lmax + 1)]

        return rl_ij, X  # 高度特征


class TensorLayerNorm(Module):
    def __init__(self, hidden_dim: int = 256, norm: str = "layer", lmax: int = 2):
        super().__init__()
        self.lmax = lmax
        self.norms = ModuleList([str2norm(norm, hidden_dim) for _ in range(lmax)])

    def forward(self, X: list):
        out_X = [self.norms[i](X[i]) for i in range(self.lmax)]
        return out_X  # [L] b n l d


class GATA(Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        lmax: int = 2,
        activation: str = "silu",
        epsilon: float = 1e-7,
        norm: str = "layer",
        steerable_norm: str = "",
        cutoff: float = 5.0,
        head_dim: int = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        edge_updates: Union[bool, str] = True,
        last_layer: bool = False,
        scale_edge: bool = True,
        evec_dim: int = None,
        emlp_dim: int = None,
        rejection: bool = True,
        gated: str = None,
        weight_init=xavier_uniform,
        bias_init=zeros_initializer,
    ):
        super().__init__()
        self.lmax = lmax
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.edge_updates = edge_updates
        self.scale_edge = scale_edge
        self.activation = activation
        multiplier = 2 * lmax + 1
        self.multiplier = multiplier
        dim_inner = head_dim * num_heads if exists(head_dim) else hidden_dim
        self.head_dim = head_dim if exists(head_dim) else hidden_dim
        self.norm = str2norm(norm, hidden_dim) if norm != "" else nn.Identity()
        self.degree_norm = TensorLayerNorm(hidden_dim, lmax=self.lmax) if steerable_norm != "" else None
        self.edge_vec_dim = hidden_dim if evec_dim is None else evec_dim
        self.edge_mlp_dim = hidden_dim if emlp_dim is None else emlp_dim
        self.rejection = rejection
        # attn
        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)
        InitMLP = partial(MLP, weight_init=weight_init, bias_init=bias_init)
        self.W_q = InitDense(hidden_dim, dim_inner, bias=False)
        self.W_k = InitDense(hidden_dim, dim_inner, bias=False)
        self.W_re = InitDense(hidden_dim, dim_inner, bias=False, activation=activation)
        self.W_rs = InitDense(hidden_dim, multiplier * dim_inner, bias=False, activation=None)
        self.gamma_s = InitMLP([hidden_dim, hidden_dim, multiplier * dim_inner], False, activation, last_activation=None)
        self.gamma_v = InitMLP([hidden_dim, hidden_dim, multiplier * dim_inner], False, activation, last_activation=None)
        self.W_vq = InitDense(hidden_dim, self.edge_vec_dim, activation=None, bias=False)
        self.W_vk = ModuleList([InitDense(hidden_dim, self.edge_vec_dim, activation=None, bias=False) for _ in range(self.lmax)])
        self.gamma_t = InitDense(hidden_dim, hidden_dim, activation=activation)
        self.gamma_w = nn.Sequential(
            LayerNorm(self.edge_vec_dim),
            str2act(activation),
            InitDense(self.edge_vec_dim, hidden_dim, activation=None),
            str2gate(gated) if exists(gated) else nn.Identity(),
        )
        self.split_heads = Rearrange("b ... (h d) -> b h ... d", h=num_heads)
        self.merge_heads = Rearrange("b h ... d -> b ... (h d)")

        self.cutoff = CosineCutoff(cutoff)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):

        self.gamma_s.reset_parameters()

        self.W_q.reset_parameters()
        self.W_k.reset_parameters()

        self.gamma_v.reset_parameters()
        self.W_re.reset_parameters()
        self.W_rs.reset_parameters()

        if not self.last_layer and self.edge_updates:
            self.gamma_t.reset_parameters()
            self.W_vq.reset_parameters()
            for w in self.W_vk:
                w.reset_parameters()

    def forward(self, h, X: list, rl_ij: list, t_ij, r_ij, neighbor_index, neighbor_mask):
        h = self.norm(h)  # b n d
        if exists(self.degree_norm):
            X = self.degree_norm(X)

        q = self.W_q(h)  # b n h d)
        k = self.W_k(h)  # b n (h d)
        x = self.gamma_s(h)  # b n (m*h d)
        v = self.gamma_v(h)  # b n (m*h d)
        t_ij_attn = self.W_re(t_ij)  # b n k d
        t_ij_filter = self.W_rs(t_ij)  # b n k (m*h d)

        (qi, kj, vj, t_ij_attn) = map(self.split_heads, (q, k, v, t_ij_attn))  # b h n ... d

        kj = get_at("b h [n] ..., b i j -> b h i j ...", kj, neighbor_index)
        vj = get_at("b h [n] ..., b i j -> b h i j ...", vj, neighbor_index)
        xj = get_at("b [n] ..., b i j -> b i j ...", x, neighbor_index)
        Xj = [get_at("b [n] ..., b i j -> b i j ...", X[i], neighbor_index) for i in range(self.lmax)]
        
        attn = einsum("... i d, ... i j d, ... i j d -> ... i j", qi, kj, t_ij_attn)  # b h n k
        attn = einx.where("b i j, b h i j, -> b h i j", neighbor_mask, attn, -1e9) # b h n k
        attn = F.softmax(attn, dim=-1)

        if self.scale_edge:
            mask = neighbor_mask.float()  # b n k
            n_edges = mask.sum(dim=-1, keepdim=True)  # b n 1
            norm = torch.sqrt(n_edges) / math.sqrt(self.head_dim)
            norm = einx.rearrange("b i j -> b 1 i j", norm)
        else:
            norm = 1.0 / math.sqrt(self.head_dim)
        attn = self.dropout(attn * norm)
        attn = einx.rearrange("... -> ... 1", attn)
        sea_ij = attn * vj  # b h n k (m*d)
        sea_ij = self.merge_heads(sea_ij)  # b n (h*m d)

        spatial_attn = t_ij_filter * xj * self.cutoff(r_ij.unsqueeze(-1))
        outputs = spatial_attn + sea_ij
        components = einx.rearrange("b i j (m d) -> m b i j 1 d", outputs, m=self.multiplier)  # m b n k 1 d
        o_s_ij, o_d_l_ij, o_t_l_ij = components[0], components[1 : self.lmax + 1], components[self.lmax + 1 :]
        dX_R = [rl_ij[i][..., None] * o_d_l_ij[i] for i in range(self.lmax)]
        dX_X = [Xj[i] * o_t_l_ij[i] for i in range(self.lmax)]
        d_X = [dX_R[i] + dX_X[i] for i in range(self.lmax)]

        d_h = reduce(o_s_ij, "b i j c d -> b i d", "sum")  # aggr
        d_X = [reduce(dX_R[i] + dX_X[i], "b i j c d -> b i c d", "sum") for i in range(self.lmax)]  # aggr
        X = [X[i] + d_X[i] for i in range(self.lmax)]
        h = h + d_h

        # htr
        if not self.last_layer and self.edge_updates:
            X_htr = X
            eq = [self.W_vq(X_htr[i]) for i in range(self.lmax)]  # l b n d
            eqi = [repeat(eq[i], "b n c d -> b n k c d", k=neighbor_index.shape[2]) for i in range(self.lmax)]
            ek = [self.W_vk[i](X_htr[i]) for i in range(self.lmax)]
            ekj = [get_at("b [n] ..., b i j -> b i j ...", ek[i], neighbor_index) for i in range(self.lmax)]
            if self.rejection:
                eqi = vector_rejection(eqi, rl_ij, self.lmax)
                ekj = vector_rejection(ekj, -rl_ij, self.lmax)  # b n k d
            w_ij = [einsum("... c d, ... c d -> ... d", eqi[i], ekj[i]) for i in range(self.lmax)]
            w_ij = [einx.where("b i j, b i j d, -> b i j d", neighbor_mask, w_ij[i], 0.0) for i in range(self.lmax)]
            w_ij = torch.add(*w_ij)  # aggr
            d_t_ij = self.gamma_t(t_ij) * self.gamma_w(w_ij)  # b n k d
            t_ij = t_ij + d_t_ij

        return h, X, t_ij  # d_h, d_X, d_t_ij


class EQFF(Module):

    def __init__(
        self,
        hidden_dim: int = 64,
        lmax: int = 2,
        activation: str = "silu",
        epsilon: float = 1e-8,
        weight_init=xavier_uniform,
        bias_init=zeros_initializer,
    ):
        super().__init__()
        self.lmax = lmax
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        context_dim = 2 * hidden_dim
        out_size = 2
        self.gamma_m = nn.Sequential(
            InitDense(context_dim, hidden_dim, activation=activation),
            InitDense(hidden_dim, out_size * hidden_dim, activation=None),
        )
        self.dim_sizes = [2 * l + 1 for l in range(1, lmax + 1)]
        self.W_vu = InitDense(hidden_dim, hidden_dim, activation=None, bias=False)

    def reset_parameters(self):
        self.W_vu.reset_parameters()
        for l in self.gamma_m:
            l.reset_parameters()

    def forward(self, h, X):
        h = einx.rearrange("b n d -> b n 1 d", h)  # b n 1 d
        X_p_cat = torch.cat(X, dim=-2)  # b n 9 d
        X_p = self.W_vu(X_p_cat)

        X_pn = torch.sqrt(torch.sum(X_p**2, dim=-2, keepdim=True) + self.epsilon)
        ctx = torch.cat([h, X_pn], dim=-1)
        x = self.gamma_m(ctx)

        m1, m2 = torch.split(x, self.hidden_dim, dim=-1)
        dX_intra = m2 * X_p
        h = h + m1
        X_p_cat = X_p_cat + dX_intra  # res
        h = einx.rearrange("b n 1 d -> b n d", h)
        X = torch.split(X_p_cat, self.dim_sizes, dim=-2)
        return h, X


# gotennet
class GotenNet(Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        lmax: int = 2,
        cutoff: float = 5.0,
        max_neighbors: int = 16,
        n_rbf: int = 16,
        max_z: int = 100,
        is_embed: bool = False,
        epsilon: float = 1e-7,
        norm: str = "layer",
        steerable_norm: str = "",
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        edge_updates: Union[bool, str] = True,
        scale_edge: bool = True,
        activation: str = "silu",
        basis: str = "gaussian",
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

        # init
        weight_init = str2init(weight_init)
        bias_init = zeros_initializer
        # weight
        self.embed = Linear(max_z, hidden_dim) if is_embed else nn.Embedding(max_z, hidden_dim)  # A_na
        radial_basis = str2basis(basis)
        self.radial_basis = radial_basis(cutoff=cutoff, n_rbf=n_rbf)

        self.node_init = NodeInit(
            hidden_dim=hidden_dim,
            n_rbf=n_rbf,
            cutoff=cutoff,
            max_z=max_z,
            is_embed=is_embed,
            activation=activation,
            proj_ln="layer",
            weight_init=weight_init,
            bias_init=bias_init,
        )
        self.edge_init = EdgeInit(hidden_dim, n_rbf, weight_init=weight_init, bias_init=bias_init)
        self.tensor_init = TensorInit(lmax)
        self.layers = ModuleList(
            [
                ModuleList(
                    [
                        GATA(
                            hidden_dim,
                            lmax,
                            activation,
                            epsilon,
                            norm,
                            cutoff=cutoff,
                            num_heads=num_heads,
                            dropout=attn_dropout,
                            edge_updates=edge_updates,
                            scale_edge=scale_edge,
                            gated="tanh",
                            weight_init=weight_init,
                            bias_init=bias_init,
                        ),
                        EQFF(
                            hidden_dim,
                            lmax,
                            activation,
                            epsilon,
                            weight_init=weight_init,
                            bias_init=bias_init,
                        ),
                    ]
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = Dense(hidden_dim, 1)

    def reset_parameters(self):
        self.node_init.reset_parameters()
        self.edge_init.reset_parameters()
        for gata, eqff in self.layers:
            gata.reset_parameters()
            eqff.reset_parameters()

    def forward(self, z: Tensor, coors: Tensor, mask: Tensor = None):
        """
        GotenNet
        Args:
            z (Tensor): [batch, nums_atoms, ] or [batch, nums_atoms, input_dim]
            coors (Tensor): [batch, nums_atoms, 3]
            mask (Tensor, optional): [batch, nums_atoms, ]. Defaults to None.
        """
        # create neighbor
        h = self.embed(z)  # b a d
        neighbor_vec, neighbor_dist, neighbor_index, neighbor_mask = create_graph(
            coors=coors, cutoff=self.cutoff, max_neighbors=self.max_neighbors, mask=mask
        )
        neighbor_rb = self.radial_basis(neighbor_dist)  # b n k r
        
        h = self.node_init(z, h, neighbor_index, neighbor_dist, neighbor_rb, neighbor_mask)  # b n d
        t_ij = self.edge_init(h, neighbor_index, neighbor_rb, neighbor_mask)  # b n k d
        rl_ij, X = self.tensor_init(h.shape, neighbor_vec)  # L b n k l X: [L] b n l d

        for gata, eqff in self.layers:
            h, X, t_ij = gata(
                h,
                X,
                rl_ij=rl_ij,
                t_ij=t_ij,
                r_ij=neighbor_dist,
                neighbor_index=neighbor_index,
                neighbor_mask=neighbor_mask,
            )
            h, X = eqff(h, X)

        h = self.out_proj(h)
        h = h.squeeze(-1)
        return h.sigmoid()


if __name__ == "__main__":
    model = GotenNet(hidden_dim=128, max_neighbors=7, num_layers=10).cuda(1)
    z = torch.randint(0, 14, (1, 9)).cuda(1)
    coords = torch.randn(1, 9, 3).cuda(1)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]).bool().cuda(1)  # [batch_size, num_atoms]
    label = torch.randint(0, 2, (1, 9)).float().cuda(1)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for _ in range(1000):
        out = model(z, coords, mask)
        print(out)
        loss = loss_fn(out, label)

        loss.backward()
        optimizer.step()
