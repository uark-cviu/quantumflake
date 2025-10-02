import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    m = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, out_dim),
    )
    nn.init.zeros_((m[-1].weight))
    nn.init.zeros_(m[-1].bias)
    return m


class CLIFFHead(nn.Module):
    def __init__(self, d_z: int, d_e: int = 128, gate_temp: float = 1.0,
                 init_materials: List[str] = None, delta_hidden: int = None):
        super().__init__()
        self.d_z = d_z
        self.d_e = d_e
        self.gate_temp = float(gate_temp)
        self.delta_hidden = delta_hidden or max(64, d_z // 2)
        self.base_head = nn.Linear(d_z, 3)
        self.materials: List[str] = []
        self.name_to_id: Dict[str, int] = {}
        self.material_embedding = nn.Embedding(max(1, len(init_materials or [])), d_e)
        self.gate_proj = nn.Linear(d_z, d_e)
        self.material_fcs = nn.ModuleDict()
        if init_materials:
            for name in init_materials:
                self.ensure_material(name)

    @property
    def num_materials(self) -> int:
        return len(self.materials)

    def ensure_material(self, name: str) -> int:
        if name in self.name_to_id:
            return self.name_to_id[name]

        mid = len(self.materials)
        self.materials.append(name)
        self.name_to_id[name] = mid

        if mid >= self.material_embedding.num_embeddings:
            self._expand_embeddings(mid + 1)

        in_dim = self.d_z + self.d_e
        head = _mlp(in_dim, self.delta_hidden, 3)

        dev = next(self.base_head.parameters()).device
        dt = next(self.base_head.parameters()).dtype
        head.to(device=dev, dtype=dt)

        self.material_fcs[f"material_{mid}"] = head
        with torch.no_grad():
            self.material_embedding.weight[mid].zero_()

        return mid

    def _expand_embeddings(self, new_n: int):
        old = self.material_embedding
        dev, dt = old.weight.device, old.weight.dtype
        new = nn.Embedding(new_n, self.d_e).to(device=dev, dtype=dt)
        with torch.no_grad():
            new.weight[:old.num_embeddings].copy_(old.weight.data)
            if new_n > old.num_embeddings:
                nn.init.normal_(new.weight[old.num_embeddings:], mean=0.0, std=0.01)
        self.material_embedding = new

    def logits_3M(self, z: torch.Tensor, scale_delta_by_gate: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.base_head(z)  # [B, 3]
        q = F.normalize(self.gate_proj(z), dim=1)                 # [B, d_e]
        E = F.normalize(self.material_embedding.weight, dim=1)    # [M, d_e]
        sims = q @ E.t()                                          # [B, M]
        w = F.softmax(sims / max(self.gate_temp, 1e-6), dim=1)    # [B, M]
        B, M = z.size(0), self.num_materials
        blocks = []
        for m in range(M):
            em = self.material_embedding.weight[m].unsqueeze(0).expand(B, -1)     # [B, d_e]
            delta = self.material_fcs[f"material_{m}"](torch.cat([z, em], dim=1)) # [B, 3]
            if scale_delta_by_gate:
                delta = w[:, m:m+1] * delta
            y_m = b + delta
            blocks.append(y_m)

        L = torch.cat(blocks, dim=1)  # [B, 3*M]
        return L, sims
