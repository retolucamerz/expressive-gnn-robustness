import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool, global_max_pool

# from models.GIN import DenseGINConv
from datasets import with_encoders
from models.GINE import DenseGINEConv, GINEConv
from datasets.metrics import RegressionEvaluator, with_metrics
from models.util import convert_to_leaky_relu, to_dense_data


def get_laplacian_dense(adj, mask=None, deg_adj=None):
    """computes the degree-normalized laplacian of a given batch of adjacency matricies, encoded as a (B, N, N) tensor `adj`"""
    B, N, _ = adj.shape
    eye = torch.eye(N).unsqueeze(0).expand(B, -1, -1).clone().to(adj.device)
    D_invsqrt = torch.zeros_like(adj)
    if deg_adj is None:
        deg_adj = adj
    deg = deg_adj.sum(dim=-1)
    deg[deg<0.001] = 1
    D_invsqrt[:, range(N), range(N)] = 1 / deg.sqrt()
    laplacian = eye - D_invsqrt @ adj @ D_invsqrt

    if mask is None:
        mask = adj.sum(dim=-1) > 0

    # force eigenval entry for non-existant nodes to be zero
    fake_entries = torch.zeros_like(adj, device=adj.device, dtype=torch.float64)
    fake_entries[:, range(N), range(N)] = (
        3
        + torch.arange(N, device=adj.device).unsqueeze(0).repeat(B, 1)
        / 5  # separate additional eigenvals by at least 1/5
    ) * (
        1 - mask.to(dtype=torch.float64)
    )  # since 3 > 2 and largest eigenvalue of laplacian <= 2
    laplacian += fake_entries
    return laplacian


# POSITIONAL ENCODINGS
class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        use_bn=False,
        use_ln=False,
        dropout=0.5,
        negative_slope=0,
        residual=False,
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if use_bn:
            self.bns = nn.ModuleList()
        if use_ln:
            self.lns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln:
                    self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        if negative_slope == 0:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError("invalid dimension of x")
            if self.use_ln:
                x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape:
                x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x


class GIN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        use_bn=True,
        dropout=0.5,
        negative_slope=0,
    ):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        if use_bn:
            self.bns = nn.ModuleList()
        self.use_bn = use_bn
        if negative_slope == 0:
            self.activation = nn.ReLU()
        else:
            self.activation = (nn.LeakyReLU(negative_slope=self.negative_slope),)
        # input layer
        update_net = MLP(
            in_channels,
            hidden_channels,
            hidden_channels,
            2,
            use_bn=use_bn,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        self.layers.append(GINConv(nn=update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                2,
                use_bn=use_bn,
                dropout=dropout,
                negative_slope=negative_slope,
            )
            self.layers.append(GINConv(nn=update_net))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(
            hidden_channels,
            hidden_channels,
            out_channels,
            2,
            use_bn=use_bn,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        self.layers.append(GINConv(nn=update_net))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        # x: (in_channels, N, 1)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i - 1](x)
                    elif x.ndim == 3:
                        x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
                    else:
                        raise ValueError("invalid x dim")
            x = layer(x, edge_index)
        return x


class GINDeepSigns(nn.Module):
    """Sign invariant neural network
    f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        k,
        use_bn=False,
        use_ln=False,
        dropout=0.5,
        negative_slope=0,
    ):
        super(GINDeepSigns, self).__init__()
        self.enc = GIN(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            use_bn=use_bn,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        rho_dim = out_channels * k
        self.rho = MLP(
            rho_dim,
            hidden_channels,
            k,
            num_layers,
            use_bn=use_bn,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        self.k = k

    def forward(self, x, edge_index):
        N, _ = x.shape

        x = x.transpose(0, 1).unsqueeze(-1)  # (in_channels, N, 1)
        x = self.enc(x.clone(), edge_index) + self.enc(
            -x.clone(), edge_index
        )  # (in_channels, N, out_channels)

        x = x.transpose(0, 1).reshape(N, -1)  # concat features, (N, rho_dim)
        x = self.rho(x)  # (N, k)
        return x


# GINE MODEL USING POSITIONAL ENCODING
class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, negative_slope=0, L=2):
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        if negative_slope == 0:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = self.activation(y)
        y = self.FC_layers[self.L](y)
        return y


@with_metrics
@with_encoders
class SignGINE(nn.Module):
    def __init__(
        self,
        num_outputs,
        out_dim=10,  # out dimension of the GNN layers (before pooling)
        hidden_dim=95,
        n_layers=16,
        in_feat_dropout=0,
        dropout=0,
        batch_norm=True,
        residual=True,
        pos_enc_dim=16,
        sign_inv_layers=8,
        phi_out_dim=4,
        readout="mean",
    ):
        super(SignGINE, self).__init__()
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.residual = residual
        self.readout = readout

        self.pos_enc_dim = pos_enc_dim
        self.sign_inv_net = GINDeepSigns(
            1,
            hidden_dim,
            phi_out_dim,
            sign_inv_layers,
            self.pos_enc_dim,
            use_bn=True,
            dropout=dropout,
        )
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList(
            [
                GINEConv(
                    nn=MLP(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        2,
                        use_bn=self.batch_norm,
                        dropout=dropout,
                    )
                )
                for _ in range(self.n_layers - 1)
            ]
        )
        self.layers.append(
            GINEConv(
                nn=MLP(
                    hidden_dim,
                    hidden_dim,
                    out_dim,
                    2,
                    use_bn=self.batch_norm,
                    dropout=dropout,
                )
            )
        )

        self.MLP_layer = MLPReadout(out_dim, num_outputs)

    def forward(self, data, **kwargs):
        # compute positional encoding
        _, mask, adj, _ = to_dense_data(data)
        laplacian = get_laplacian_dense(adj, mask=mask)
        e, v = torch.linalg.eigh(laplacian, UPLO="L")

        # discard eigenvec with eigenval 0 and pad
        n_eigenvecs = v.shape[-1]
        if n_eigenvecs <= self.pos_enc_dim:
            pad = (0, self.pos_enc_dim - n_eigenvecs + 1)
            v_k = F.pad(v[:, :, 1:], pad=pad, mode="constant", value=0)
        else:
            v_k = v[:, :, 1 : self.pos_enc_dim + 1]

        pos_enc = self.sign_inv_net(v_k[mask], data.edge_index)
        pos_enc = self.embedding_p(pos_enc)
        x = data.x + pos_enc

        # usual GINE
        for conv in self.layers:
            x = conv(x, data.edge_index, data.edge_attr)

        if self.readout == "mean":
            x = global_mean_pool(x, data.batch)
        elif self.readout == "add":
            x = global_add_pool(x, data.batch)
        elif self.readout == "max":
            x = global_max_pool(x, data.batch)
        else:
            raise ValueError(f"invalid readout method '{self.readout}'")

        return self.MLP_layer(x)


# DENSE MODEL
class DenseGINConv(torch.nn.Module):
    """copied from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/dense_gin_conv.html#DenseGINConv,
    adjusted to apply `self.nn` to a masked view of `x` which is equivalent to the data in the sparse version
    """

    @staticmethod
    def from_sparse(sparse_gine_conv, negative_slope=0):
        conv = DenseGINConv(nn=None)
        conv.nn = convert_to_leaky_relu(sparse_gine_conv.nn, negative_slope)
        conv.eps = sparse_gine_conv.eps
        conv.initial_eps = sparse_gine_conv.initial_eps
        return conv

    def __init__(
        self,
        nn: Module,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Tensor,
        add_loop: bool = True,
    ) -> Tensor:
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        orig_shape = x.shape
        x = x.view(*x.shape[:2], -1)
        out = torch.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out
        out = out.view(*orig_shape)

        nn_out = self.nn(out[mask].transpose(0, 1)).transpose(0, 1)
        out = torch.zeros((*orig_shape[:2], *nn_out.shape[1:]), device=x.device)
        out[mask] = nn_out

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


class DenseGIN(nn.Module):
    @staticmethod
    def from_sparse(sparse_gine, negative_slope=0):
        dense = DenseGIN(
            in_channels=1,
            hidden_channels=1,
            out_channels=1,
            n_layers=1,
            negative_slope=negative_slope,
        )

        dense.use_bn = sparse_gine.use_bn
        dense.activation = sparse_gine.activation
        dense.dropout = sparse_gine.dropout

        dense.bns = sparse_gine.bns
        dense.layers = nn.ModuleList()
        for conv in sparse_gine.layers:
            dense.layers.append(DenseGINConv.from_sparse(conv, negative_slope))

        return dense

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        use_bn=True,
        dropout=0.5,
        negative_slope=0,
    ):
        super(DenseGIN, self).__init__()
        self.layers = nn.ModuleList()
        if use_bn:
            self.bns = nn.ModuleList()
        self.use_bn = use_bn

        if negative_slope == 0:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(negative_slope=negative_slope)

        # input layer
        update_net = MLP(
            in_channels,
            hidden_channels,
            hidden_channels,
            2,
            use_bn=use_bn,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        self.layers.append(DenseGINConv(nn=update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                2,
                use_bn=use_bn,
                dropout=dropout,
                negative_slope=negative_slope,
            )
            self.layers.append(DenseGINConv(nn=update_net))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(
            hidden_channels,
            hidden_channels,
            out_channels,
            2,
            use_bn=use_bn,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        self.layers.append(DenseGINConv(nn=update_net))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj, mask):
        # (B, N, in_channels, 1)
        B, N, _, _ = x.shape
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim < 4:
                        raise ValueError("x has too few dimensions")
                    elif x.ndim == 4:
                        # (n, chan, 1) -> (chan, 1, n) -> (n, chan, 1)
                        x[mask] = self.bns[i - 1](x[mask].permute(1, 2, 0)).permute(
                            2, 0, 1
                        )
                        # (B, N, in_channels, 1)
                    else:
                        raise ValueError("invalid x dim")

            x = layer(x, adj, mask)

        return x  # (B, N, in_channels, out_channels)


class DenseGINDeepSigns(nn.Module):
    """Sign invariant neural network
    f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    @staticmethod
    def from_sparse(sparse_signs, negative_slope=0):
        dense_signs = DenseGINDeepSigns(1, 1, 1, 1, 1)
        dense_signs.enc = DenseGIN.from_sparse(
            sparse_signs.enc, negative_slope=negative_slope
        )
        dense_signs.rho = sparse_signs.rho
        dense_signs.k = sparse_signs.k
        return dense_signs

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        k,
        use_bn=False,
        use_ln=False,
        dropout=0.5,
        negative_slope=0,
    ):
        super(DenseGINDeepSigns, self).__init__()
        self.enc = DenseGIN(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            use_bn=use_bn,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        rho_dim = out_channels * k
        self.rho = MLP(
            rho_dim,
            hidden_channels,
            k,
            num_layers,
            use_bn=use_bn,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        self.k = k

    def forward(self, x, mask, adj):
        B, N, _ = x.shape
        # x: (B, N, in_channels)
        x = x.unsqueeze(-1)  # (B, N, in_channels, 1)
        x = self.enc(x.clone(), adj, mask) + self.enc(
            -x.clone(), adj, mask
        )  # (B, N, in_channels, out_channels)
        x = x.reshape(B, N, -1)  # concat features, (B, N, rho_dim)

        rho_out = self.rho(x[mask])
        x = torch.zeros((*x.shape[:2], *rho_out.shape[1:]), device=rho_out.device)
        x[mask] = rho_out
        return x  # (B, N, k)


@with_metrics
@with_encoders
class DenseSignGINE(nn.Module):
    @staticmethod
    def from_sparse(sparse_signgine, negative_slope=0):
        atom_encoder = (
            sparse_signgine.atom_encoder if sparse_signgine.use_atom_embedding else None
        )
        bond_encoder = (
            sparse_signgine.bond_encoder if sparse_signgine.use_bond_embedding else None
        )

        dense_signgine = DenseSignGINE(
            1,
            out_dim=4,
            hidden_dim=1,
            n_layers=1,
            pos_enc_dim=1,
            sign_inv_layers=1,
            phi_out_dim=1,
            evaluator=sparse_signgine.evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
        )

        dense_signgine.n_layers = sparse_signgine.n_layers
        dense_signgine.batch_norm = sparse_signgine.batch_norm
        dense_signgine.residual = sparse_signgine.residual
        dense_signgine.readout = sparse_signgine.readout
        dense_signgine.pos_enc_dim = sparse_signgine.pos_enc_dim
        dense_signgine.embedding_p = sparse_signgine.embedding_p
        dense_signgine.in_feat_dropout = sparse_signgine.in_feat_dropout
        dense_signgine.embedding_p = sparse_signgine.embedding_p

        dense_signgine.sign_inv_net = DenseGINDeepSigns.from_sparse(
            sparse_signgine.sign_inv_net, negative_slope=negative_slope
        )
        dense_signgine.layers = nn.ModuleList()
        for conv in sparse_signgine.layers:
            dense_signgine.layers.append(
                DenseGINEConv.from_sparse(conv, negative_slope=negative_slope)
            )
        dense_signgine.MLP_layer = sparse_signgine.MLP_layer
        dense_signgine.MLP_layer.activation = nn.LeakyReLU(
            negative_slope=negative_slope
        )

        return dense_signgine

    def __init__(
        self,
        num_outputs,
        out_dim=10,  # out dimension of the GNN layers (before pooling)
        hidden_dim=95,
        n_layers=16,
        in_feat_dropout=0,
        dropout=0,
        batch_norm=True,
        residual=True,
        pos_enc_dim=16,
        sign_inv_layers=8,
        phi_out_dim=4,
        negative_slope=0,
        readout="mean",
    ):
        super(DenseSignGINE, self).__init__()
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.residual = residual
        self.readout = readout

        if pos_enc_dim<=0:
            raise ValueError("pos enc dim needs to be positive")

        self.pos_enc_dim = pos_enc_dim
        self.sign_inv_net = DenseGINDeepSigns(
            1,
            hidden_dim,
            phi_out_dim,
            sign_inv_layers,
            self.pos_enc_dim,
            use_bn=True,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList(
            [
                DenseGINEConv(
                    nn=MLP(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        2,
                        use_bn=self.batch_norm,
                        dropout=dropout,
                        negative_slope=negative_slope,
                    )
                )
                for _ in range(self.n_layers - 1)
            ]
        )
        self.layers.append(
            DenseGINEConv(
                nn=MLP(
                    hidden_dim,
                    hidden_dim,
                    out_dim,
                    2,
                    use_bn=self.batch_norm,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )
        )

        self.MLP_layer = MLPReadout(out_dim, num_outputs, negative_slope=negative_slope)

    def forward(self, data, x=None, adj=None, edge_attr=None, mask=None, noisy=None):
        x, mask, adj, edge_attr = to_dense_data(data, x=x, adj=adj, edge_attr=edge_attr, mask=mask)

        # POSITIONAL ENCODING
        adj_lap = adj.clone()
        if noisy is not None:
            adj_lap += noisy * torch.diag_embed(torch.randn(adj.shape[:2], device=adj.device))
            # adj_lap += 0.01 * torch.diag_embed(torch.randn(adj.shape[:2], device=adj.device))
        laplacian = get_laplacian_dense(adj_lap, mask=mask, deg_adj=adj)
        e, v = torch.linalg.eigh(laplacian)

        # discard eigenvec with eigenval 0 and pad
        n_eigenvecs = v.shape[-1]
        if n_eigenvecs <= self.pos_enc_dim:
            pad = (0, self.pos_enc_dim - n_eigenvecs + 1)
            v_k = F.pad(v[:, :, 1:], pad=pad, mode="constant", value=0)
        else:
            v_k = v[:, :, 1 : self.pos_enc_dim + 1]

        pos_enc = self.sign_inv_net(v_k, mask, adj)  # (B, N, k)
        pos_enc = self.embedding_p(pos_enc)  # (B, N, hidden_dim)
        x = x + pos_enc

        # GINE with PE features
        for conv in self.layers:
            x = conv(x, adj, edge_attr, mask)  # (B, N, hidden_dim)

        if self.readout == "mean":
            num_nodes = mask.sum(dim=-1)
            x = x.sum(1) / num_nodes.unsqueeze(-1)
        elif self.readout == "add":
            x = x.sum(1)
        elif self.readout == "max":
            x = x.max(1)
        else:
            raise ValueError(f"invalid readout method '{self.readout}'")

        return self.MLP_layer(x)
