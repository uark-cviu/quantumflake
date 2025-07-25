import torch
import torch.nn as nn
from transformers import ResNetModel

CLASS_NAMES = ['1-layer', '5-layer']

#Example
MATERIAL_MAP = {
    'graphene': 0, 
    'mos2': 1, 
    'wse2': 2, 
    'hbn': 3
} 


class FlakeLayerClassifier(nn.Module):
    def __init__(self, num_materials, material_dim, num_classes=2, dropout_prob=0.1, freeze_cnn=False):
        super().__init__()
        
        self.cnn = ResNetModel.from_pretrained("microsoft/resnet-18")
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        img_feat_dim = self.cnn.config.hidden_sizes[-1]
        self.material_embedding = nn.Embedding(num_materials, material_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.fc_img = nn.Sequential(
            nn.Linear(img_feat_dim, img_feat_dim),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(img_feat_dim, num_classes)
        )
        
        combined_dim = img_feat_dim + material_dim
        self.fc_comb = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(combined_dim, num_classes)
        )

    def forward(self, pixel_values, material=None):
        outputs = self.cnn(pixel_values=pixel_values)
        img_feats = outputs.pooler_output
        img_feats = img_feats.view(img_feats.size(0), -1)

        if material is None:
            return self.fc_img(img_feats)
        
        material_embeds = self.material_embedding(material)
        combined_feats = torch.cat((img_feats, material_embeds), dim=1)
        return self.fc_comb(combined_feats)

