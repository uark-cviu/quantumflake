import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer as TimmViT, _cfg
from timm.models.registry import register_model


class VisionTransformer(TimmViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Identity()

    def forward(self, x, prompt=None):
        out = self.forward_features(x, prompt=prompt)
        out['logits'] = out['pre_logits']
        return out

    def forward_features(self, x, prompt=None):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        if prompt is not None:
            p_out = prompt(x)
            x_with_prompts = p_out['prompted_embedding']  # [B, P+N, D]
            cls_tokens_with_pos = cls_tokens + self.pos_embed[:, :1, :]
            prompts = x_with_prompts[:, :prompt.length, :]                  # [B, P, D]
            patches = x_with_prompts[:, prompt.length:, :] + self.pos_embed[:, 1:, :]  # [B, N, D]
            x = torch.cat((cls_tokens_with_pos, prompts, patches), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_token_features = x[:, 0]  # [B, D]
        return {'x': x, 'pre_logits': cls_token_features}


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        try:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"], strict=False)
            print("Loaded DeiT-B weights.")
        except Exception as e:
            print("Warning: could not load pretrained weights, using random init.", e)
    return model
