"""Pytorch models"""
from typing import Optional, Tuple, Type, Sequence, Any
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.parallel import DataParallel
from cryodrgn import fft, lie_tools, utils
import cryodrgn.config
from cryodrgn.lattice import Lattice

Norm = Sequence[Any]  # mean, std


class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(
        self,
        lattice: Lattice,
        qlayers: int,
        qdim: int,
        players: int,
        pdim: int,
        in_dim: int,
        zdim: int = 1,
        encode_mode: str = "resid",
        enc_mask=None,
        enc_type="linear_lowf",
        enc_dim=None,
        domain="fourier",
        activation=nn.ReLU,
        feat_sigma: Optional[float] = None,
    ):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        self.encoder = Encoder()
        self.decoder = Decoder()


    def encode(self, *img):
        # TODO: encode
        z = self.encoder(*img)
        return z

    def cat_z(self, coords, z) -> Tensor:
        """
        coords: Bx...x3
        z: Bxzdim
        """
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat((coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, z=None) -> torch.Tensor:
        """
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        """
        decoder = self.decoder
        assert isinstance(decoder, nn.Module)
        retval = decoder(self.cat_z(coords, z) if z is not None else coords)
        return retval


    def forward(self, *args, **kwargs):
        return self.decode(*args, **kwargs)



class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        # TODO: design a proper encoder, to extract
        #  the latent code z from 2D projections

    def forward(self,):
        # TODO: define the forward process,
        #  how to get z in VAE?
        return z


class Decoder(nn.Module):
    def __init__(self,):
        super(Decoder, self).__init__()
        # TODO: design a proper decoder, to extract
        #  the latent code z from 2D projections

    def positional_encoding(self, coords: Tensor) -> Tensor:
        # TODO: encode the coordinates, https://arxiv.org/abs/2003.08934
        return encoded_coords

    def forward(self, encoded_coords, z):
        # TODO: define the forward process,
        #  input (z, coords), output voxels
        return voxels


# class SO3reparameterize(nn.Module):
#     """Reparameterize R^N encoder output to SO(3) latent variable"""
#
#     def __init__(self, input_dims, nlayers: int, hidden_dim: int):
#         super().__init__()
#         if nlayers is not None:
#             self.main = ResidLinearMLP(input_dims, nlayers, hidden_dim, 9, nn.ReLU)
#         else:
#             self.main = MyLinear(input_dims, 9)
#
#         # start with big outputs
#         # self.s2s2map.weight.data.uniform_(-5,5)
#         # self.s2s2map.bias.data.uniform_(-5,5)
#
#     def sampleSO3(
#         self, z_mu: torch.Tensor, z_std: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Reparameterize SO(3) latent variable
#         # z represents mean on S2xS2 and variance on so3, which enocdes a Gaussian distribution on SO3
#         # See section 2.5 of http://ethaneade.com/lie.pdf
#         """
#         # resampling trick
#         if not self.training:
#             return z_mu, z_std
#         eps = torch.randn_like(z_std)
#         w_eps = eps * z_std
#         rot_eps = lie_tools.expmap(w_eps)
#         # z_mu = lie_tools.quaternions_to_SO3(z_mu)
#         rot_sampled = z_mu @ rot_eps
#         return rot_sampled, w_eps
#
#     def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
#         z = self.main(x)
#         z1 = z[:, :3].double()
#         z2 = z[:, 3:6].double()
#         z_mu = lie_tools.s2s2_to_SO3(z1, z2).float()
#         logvar = z[:, 6:]
#         z_std = torch.exp(0.5 * logvar)  # or could do softplus
#         return z_mu, z_std
