import torch
import torch.nn as nn


class Prompt(nn.Module):
    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key='mean',
        prompt_init='uniform',
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init='uniform',
    ):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key_enabled = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            else:
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        else:
            self.prompt = nn.Parameter(torch.zeros((1, length, embed_dim)))

        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            else:
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            self.prompt_key = None

    @staticmethod
    def l2_normalize(x, dim=None, eps=1e-12):
        sq = torch.sum(x ** 2, dim=dim, keepdim=True)
        inv = torch.rsqrt(torch.clamp(sq, min=eps))
        return x * inv

    def forward(self, x_embed):
        B = x_embed.shape[0]
        out = {}

        if not self.prompt_pool:
            out['prompted_embedding'] = torch.cat(
                [self.prompt.expand(B, -1, -1), x_embed], dim=1)
            return out

        if self.embedding_key == 'mean':
            x_key = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_key = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_key = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            x_key = torch.mean(x_embed, dim=1)

        if self.prompt_key_enabled and self.prompt_key is not None:
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)
            x_norm = self.l2_normalize(x_key, dim=1)
            sim = torch.matmul(x_norm, prompt_norm.t())  # [B, pool_size]

            if self.top_k and self.top_k > 0:
                top_k_sim, top_k_idx = torch.topk(sim, k=self.top_k, dim=1)  # [B, k]
                out['similarity'] = sim
                out['top_k_sim'] = top_k_sim
                out['top_k_idx'] = top_k_idx
                idx = top_k_idx[:, 0]
            else:
                idx = sim.argmax(dim=1)
        else:
            idx = torch.randint(0, self.pool_size, (B,), device=x_embed.device)

        batched_prompt = self.prompt[idx, :, :]  # [B, P, D]
        out['prompt_idx'] = idx
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)  # [B, P+N, D]
        return out
