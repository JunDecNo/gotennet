### The unofficial implementation of GotenNet

Based on [gotennet-pytorch](https://github.com/lucidrains/gotennet-pytorch.git) and official repo [GotenNet](https://github.com/sarpaykent/GotenNet.git)

There may still be some issues that have not been fully verified.



#### Node Scalar Feature Initialization

$$\mathbf{m}_i=\sum_{j\in\mathcal{N}(i)}\boldsymbol{z}_j\mathbf{A}_{\mathrm{nbr}}\circ\left(\varphi(\tilde{\boldsymbol{r}}_{ij}^{(0)})\mathbf{W}_{\mathrm{ndp}}\circ\phi(\tilde{\boldsymbol{r}}_{ij}^{(0)})\right)$$

$$\boldsymbol{\underset{i,\mathrm{init}}{h}}=\sigma\left(\mathrm{LN}\left((\boldsymbol{\underset i{z}}\mathbf{A_{na}},\mathbf{m_i})\mathbf{W_{nrd}}\right)\right)\mathbf{W_{nru}}$$

> official

```yaml
Args: hidden_channels, num_rbf, cutoff, max_z=100, activation=F.silu, proj_ln=''
forward: z, h, edge_index, r0_ij, varphi_r0_ij
```

```python
h = self.A_na(atomic_numbers)[:]
phi_r0_ij = self.radial_basis(edge_diff)
# NodeInit
h_src = self.A_nbr(z) 
phi_r0_ij = self.cutoff(r0_ij)
r0_ij_feat = self.W_ndp(varphi_r0_ij) * phi_r0_ij.view(-1, 1)
m_i = self.propagate(edge_index, h_src=h_src, r0_ij_feat=r0_ij_feat, size=None)
return self.W_nrd_nru(torch.cat([h, m_i], dim=1))
```

> einx

```yaml
Args: hidden_channels, num_rbf, cutoff, max_z=100, activation=F.silu, proj_ln=''
forward: z, h, neighbor_index, neighbor_dist, neighbor_rb, neighbor_mask
```

```python
h = self.embed(z)
neighbor_rb = self.radial_basis(neighbor_dist)
# NodeInit
h_src = self.embed(z)
dist_cutoff = self.cutoff(neighbor_dist)
r0_ij_feat = self.W_ndp(neighbor_rb) * dist_cutoff.unsqueeze(-1)
neighbor_feat = get_at("b [n] d, b i j -> b i j d", h_src, neighbor_index)
neighbor_feat = einx.where("b i j, b i j d, -> b i j d", neighbor_mask, neighbor_feat, 0.0) 
m_i = einsum("b i j d, b i j d -> b i d", neighbor_feat, r0_ij_feat) # aggr
return self.W_nrd_nru(torch.cat((h, m_i), dim=-1))
```

> right!!!

#### Edge Scalar Feature Initialization

$$\boldsymbol{t}_{ij,\mathrm{init}}=(\boldsymbol{h}_{i,\mathrm{init}}+\boldsymbol{h}_{j,\mathrm{init}})\circ\left(\varphi(\tilde{\boldsymbol{r}}^{(0)}{}_{ij})\mathbf{W}_{\mathrm{erp}}\right)$$

> official

```yaml
Args: num_rbf, hidden_channels, activation=None
forward: edge_index, phi_r0_ij, h
```

```python
(h_i + h_j) * self.W_erp(phi_r0_ij) # message
out = self.propagate(edge_index, h=h, phi_r0_ij=phi_r0_ij)
return out
```

> einx

```yaml
Args: hidden_dim, n_rbf, activation
forward: h, neighbor_index, neighbor_rb, neighbor_mask
```

```python
hj = get_at("b [n] d, b i j -> b i j d", h, neighbor_index)
h_ij = einx.add("b i d, b i j d -> b i j d", h, hj)
r _ij = self.W_erp(neighbor_rb) 
return h_ij * self.activation(r_ij) # currently unused
```

> right!!!

#### High-degree Steerable Feature Initialization

$$\begin{aligned}&\{\boldsymbol{o}_{ij,\mathrm{init}}^{(l)}\}_{l=1}^{L_{\mathrm{max}}}=\mathrm{split}\left(\mathbf{sea}_{ij}+(\boldsymbol{t}_{ij,\mathrm{init}}\mathbf{W}_{rs,\mathrm{init}})\circ\gamma_{s}(\boldsymbol{h}_{j,\mathrm{init}})\circ\phi(\tilde{\boldsymbol{r}}_{ij}^{(0)}),d_{ne}\right)\\&\tilde{\boldsymbol{X}}_{i,\mathrm{init}}^{(l)}=\bigoplus_{j\in\mathcal{N}(i)}\left(\boldsymbol{o}_{ij,\mathrm{init}}^{(l)}\circ\tilde{\boldsymbol{r}}_{ij}^{(l)}\right),\end{aligned}$$

Notice: This module is officially implemented in GATA. Before inputting to GATA, `X` is 0. I think `X_init` is the relevant output of the first layer of GATA, and this is also the reason for the arrow of the first layer in `Figure2(a)`.

`rl_ij`: `e3nn spherical_harmonics`

#### HIERARCHICAL TENSOR REFINEMENT (HTR)

$$\widehat{EQ_i}^{(l)}=\tilde{X}_i^{(l)}\mathbf{W}_{vq},\quad\widehat{EK}_j^{(l)}=\tilde{X}_j^{(l)}\mathbf{W}_{vk}^{(l)},\quad\mathrm{for~}l\in\{1,\ldots,L_{\max}\}$$

$$w_{ij}=\mathrm{Agg}_{l=1}^{L_{\max}}\left((\widetilde{EQ}_i^{(l)})^\top\widetilde{EK}_j^{(l)}\right)$$

Notice: This module is officially implemented in GATA. As part of the edge update.



#### GEOMETRY-AWARE TENSOR ATTENTION (GATA)

This is the most important submodule in GotenNet.

$$q_i=h_i\mathbf{W}_q,\quad k_j=\boldsymbol{h}_j\mathbf{W}_k,\quad\boldsymbol{v}_j=\gamma_v(\boldsymbol{h}_j)$$
$$\mathbf{sea}_{ij}=\frac{\exp(\alpha_{ij})}{\sum_{k\in\mathcal{N}(i)}\exp(\alpha_{ik})}v_j,\quad\mathrm{where}\quad\alpha_{ij}=q_i(k_j\circ\sigma_k(t_{ij}\mathbf{W}_{re}))^\mathrm{T}$$

$$o_{ij}^s,\{o_{ij}^{d,(l)}\}_{l=1}^{L_{\max}},\{o_{ij}^{t,(l)}\}_{l=1}^{L_{\max}}=\mathrm{split}(\mathbf{sea}_{ij}+(t_{ij}\mathbf{W}_{rs})\circ\gamma_s(\boldsymbol{h}_j)\circ\phi(\tilde{r}_{ij}^{(0)}),d_{ne})$$

$$\Delta\boldsymbol{h}_i=\bigoplus_{j\in\mathcal{N}(i)}(\boldsymbol{o}_{ij}^s),\quad\Delta\tilde{\boldsymbol{X}}_i^{(l)}=\bigoplus_{j\in\mathcal{N}(i)}\left(\boldsymbol{o}_{ij}^{d,(l)}\circ\tilde{\boldsymbol{r}}_{ij}^{(l)}+\boldsymbol{o}_{ij}^{t,(l)}\circ\tilde{\boldsymbol{X}}_j^{(l)}\right)$$

> official

```yaml
Args: ...
forward: edge_index, h, X, rl_ij, t_ij, r_ij, n_edges
```

```python
attn = (q_i * k_j * t_ij_attn).sum(dim=-1, keepdim=True)
attn = softmax(attn, index, ptr, dim_size)

sea_ij = attn * v_j.reshape(-1, self.num_heads, (self.n_atom_basis*self.multiplier) // self.num_heads)
sea_ij = sea_ij.reshape(-1, 1, self.n_atom_basis*self.multiplier)
spatial_attn = t_ij_filter.unsqueeze(1) * x_j * self.cutoff(r_ij.unsqueeze(-1).unsqueeze(-1))

dX_R = o_d_ij * rl_ij[..., None]
dX_X = o_t_ij * X_j

w_l = (EQ_i_l * EK_j_l).sum(dim=1)
```

> einx

```yaml
Args: ...
forward: h, X, rl_ij, t_ij, r_ij(neighbor_dist), neighbor_index, neighbor_mask
```

```python
attn = einsum("... i d, ... i j d, ... i j d -> ... i j", qi, kj, t_ij_attn)
attn = F.softmax(attn, dim=-1)

sea_ij = attn * vj  # b h n k (m*d)
sea_ij = self.merge_heads(sea_ij)  # b n (h*m d)
spatial_attn = t_ij_filter * xj * self.cutoff(r_ij.unsqueeze(-1))

dX_R = [rl_ij[i][..., None] * o_d_l_ij[i] for i in range(self.lmax)]
dX_X = [Xj[i] * o_t_l_ij[i] for i in range(self.lmax)]

d_h = reduce(o_s_ij, "b i j c d -> b i d", "sum")  # aggr
d_X = [reduce(dX_R[i] + dX_X[i], "b i j c d -> b i c d", "sum") for i in range(self.lmax)]  # aggr

einsum("... c d, ... c d -> ... d", eqi[i], ekj[i])
```

> right!!!



#### EQUIVARIANT FEED-FORWARD (EQFF) 



> official

```yaml
Args: hidden_dim, lmax, activation, epsilon,
forward: h, X
```

```python
X_p = self.W_vu(X)

X_pn = torch.sqrt(torch.sum(X_p**2, dim=-2, keepdim=True) + self.epsilon)

channel_context = [h, X_pn]
ctx = torch.cat(channel_context, dim=-1)

x = self.gamma_m(ctx)

m1, m2 = torch.split(x, self.n_atom_basis, dim=-1)
dX_intra = m2 * X_p

h = h + m1
X = X + dX_intra

return h, X
```

> einx

```yaml
Args: hidden_dim, lmax, activation, epsilon,
forward: h, X
```

```python
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
```

> right!!!
