import torch
import torch.nn as nn

import colorpy.ciexyz

from material_index import (
    MATERIAL_INDEX_DICT,
    Air_Index,
    MoS2_Index,
    WS2_Index,
    SiO2_Index,
    Si_Index
)

import numpy as np


class SpectrumInv(nn.Module):
    def __init__(self, start_wl, end_wl, n_wl):
        super().__init__()

        self.start_wl = start_wl
        self.end_wl = end_wl
        self.n_wl = n_wl

        self.conv = nn.Conv2d(in_channels=3, out_channels=n_wl, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Spectrum(nn.Module):
    def __init__(self, start_wl, end_wl, n_wl):
        super().__init__()
        
        self.start_wl = start_wl
        self.end_wl = end_wl
        self.n_wl = n_wl
        
        self.conv = nn.Conv2d(in_channels=n_wl, out_channels=3, kernel_size=1, stride=1, bias=False)
    
        for p in self.parameters():
            p.requires_grad = False

        for i, wl in enumerate(np.linspace(start_wl, end_wl, n_wl)):
            color = colorpy.ciexyz.xyz_from_wavelength(wl)
            color = torch.from_numpy(color)

            self.conv.weight.data[:, i, 0, 0] = color

    def forward(self, x):
        return self.conv(x)


class ShiftModule(nn.Module):
    def __init__(self, start_wl, end_wl, n_wl):
        super().__init__()

        self.start_wl = start_wl
        self.end_wl = end_wl
        self.n_wl = n_wl

        self.spectrum_inv = SpectrumInv(start_wl, end_wl, n_wl)

        self.M_inv = nn.Parameter(torch.randn(n_wl, 2, 2))
        self.W_inv = nn.Parameter(torch.randn(n_wl, 1, 2))

        self.W = nn.Parameter(torch.linalg.pinv(self.W_inv.data))
        self.W.data += torch.randn(n_wl, 2, 1) * 0.01
        self.M = nn.Parameter(torch.linalg.pinv(self.M_inv.data))
        self.M.data += torch.randn(n_wl, 2, 2) * 0.01

        self.spectrum = Spectrum(start_wl, end_wl, n_wl)

        self.init_weight()

    def random(self, scale=0.01):
        self.M.data += torch.randn(self.n_wl, 2, 2).to(self.M.data.device) * scale
        self.W.data += torch.randn(self.n_wl, 2, 1).to(self.W.data.device) * scale

    def change(self):
        for i, wl in enumerate(np.linspace(self.start_wl, self.end_wl, self.n_wl)):
            n_mat = [Air_Index(wl), WS2_Index(wl), SiO2_Index(wl), Si_Index(wl)]

            ts, rs = [], []

            Ms = []

            for j in range(3):
                ts.append(self.compute_t(n_mat[j], n_mat[j+1]))
                rs.append(self.compute_r(n_mat[j], n_mat[j+1]))

                Ms.append(torch.tensor([[1, rs[j]], [rs[j], 1]]) / ts[j])

            self.M.data[i] = Ms[0]

            W = Ms[1]
            P = torch.tensor([[np.exp(-285j), 0], [0, np.exp(285j)]]).to(torch.float32)
            W = torch.matmul(W, P)
            W = torch.matmul(W, Ms[2])
            W = torch.matmul(W, torch.tensor([[1.], [0.]]))

            self.W.data[i] = W

    def compute_t(self, n_1, n_2):
        n_1 = torch.tensor(n_1).to(torch.float32)
        n_2 = torch.tensor(n_2).to(torch.float32)
        return 2 * n_1 / (n_1 + n_2)

    def compute_r(self, n_1, n_2):
        n_1 = torch.tensor(n_1).to(torch.float32)
        n_2 = torch.tensor(n_2).to(torch.float32)
        return (n_1 - n_2) / (n_1 + n_2)

    def set_train_inv(self):
        for p in self.M_inv.parameters():
            p.requires_grad = True

        for p in self.W_inv.parameters():
            p.requires_grad = True

        for p in self.M.parameters():
            p.requires_grad = False

        for p in self.W.parameters():
            p.requires_grad = False
        
        for p in self.spectrum_inv.parameters():
            p.requires_grad = False
        
        for p in self.spectrum.parameters():
            p.requires_grad = False

    def init_weight(self):
        for i, wl in enumerate(np.linspace(self.start_wl, self.end_wl, self.n_wl)):
            n_mat = [Air_Index(wl), MoS2_Index(wl), SiO2_Index(wl), Si_Index(wl)]

            ts, rs = [], []

            Ms = []

            for j in range(3):
                ts.append(self.compute_t(n_mat[j], n_mat[j+1]))
                rs.append(self.compute_r(n_mat[j], n_mat[j+1]))

                Ms.append(torch.tensor([[1, rs[j]], [rs[j], 1]]) / ts[j])

            self.M.data[i] = Ms[0]

            W = Ms[1]
            P = torch.tensor([[np.exp(-285j), 0], [0, np.exp(285j)]]).to(torch.float32)
            W = torch.matmul(W, P)
            W = torch.matmul(W, Ms[2])
            W = torch.matmul(W, torch.tensor([[1.], [0.]]))

            self.W.data[i] = W

        self.M_inv.data = torch.linalg.pinv(self.M.data)
        self.W_inv.data = torch.linalg.pinv(self.W.data)

    def forward(self, x):

        x = self.spectrum_inv(x)

        a = torch.ones_like(x)
        x = torch.stack([a, x], dim=1) # (B, 2, D, H, W)
        x = x.unsqueeze(dim=2) # (B, 2, 1, D, H, W)

        x = torch.einsum('dij,bjkdhw->bikdhw', self.M_inv, x) # (B, 2, 1, D, H, W)
        x = torch.einsum('bikdhw,dkj->bijdhw', x, self.W_inv) # (B, 2, 2, D, H, W)

        x = torch.einsum('bijdhw,djk->bikdhw', x, self.W) # (B, 2, 2, D, H, W)
        x = torch.einsum('dji,bikdhw->bjkdhw', self.M, x) # (B, 2, 1, D, H, W)

        b = x[:, 0, 0]
        x = x[:, 1, 0] # (B, D, H, W)

        x = self.spectrum(x)

        return x
        

if __name__=='__main__':
    shift_module = ShiftModule(400, 760, 160)

    t_input = torch.randn(1, 3, 224, 224)
    t_output = shift_module(t_input)
    print(t_output.shape)
