
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lib.xttn import mask_xattn_one_text
from typing import Optional

def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n

# Convolutional Module for CAA
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        # Convolution Layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # Normalization Layer
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # Activation Layer
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # Combine all layers
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

# Context Anchor Attention (CAA) Module
class CAA(nn.Module):
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels, norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels, norm_cfg=None, act_cfg=None)
        # Ensure the final convolution reduces the channel dimension to 1
        self.conv2 = ConvModule(channels, 1, 1, 1, 0, norm_cfg=None, act_cfg=None)  # Output channel is set to 1
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Apply the convolutional operations and activation
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        # Ensure the output has a single channel
        if attn_factor.shape[1] != 1:
            raise ValueError(f"Expected channel dimension to be 1, but got {attn_factor.shape[1]}")
        return attn_factor


# Token Sparse Module
class TokenSparse(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()        
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio    

    def forward(self, tokens, attention_x, attention_y):
        B_v, L_v, C = tokens.size()
        score = attention_x + attention_y        
        num_keep_token = math.ceil(L_v * self.sparse_ratio)    
        score_sort, score_index = torch.sort(score, dim=1, descending=True)        
        keep_policy = score_index[:, :num_keep_token]
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)        
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))
        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))   
        non_keep_score = score_sort[:, num_keep_token:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True) 
        return select_tokens, extra_token, score_mask                 

# Token Aggregation Module
class TokenAggregation(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()        
        hidden_dim = int(dim * dim_ratio)
        self.weight = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, keeped_patches)
                        )        
        self.scale = nn.Parameter(torch.ones(1, 1, 1))       

    def forward(self, x, keep_policy=None):
        weight = self.weight(x)
        weight = weight.transpose(2, 1) * self.scale       
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)
            weight = weight - (1 - keep_policy) * 1e10
        weight = F.softmax(weight, dim=2)
        x = torch.bmm(weight, x)        
        return x

# Cross Sparse Aggregation Network v2
class CrossSparseAggrNet_v2(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt      
        self.hidden_dim = opt.embed_size  
        self.num_patches = opt.num_patches
        self.sparse_ratio = opt.sparse_ratio  
        self.aggr_ratio = opt.aggr_ratio  
        self.attention_weight = opt.attention_weight
        self.ratio_weight = opt.ratio_weight        
        self.keeped_patches = int(self.num_patches * self.aggr_ratio * self.sparse_ratio)
        self.sparse_net = TokenSparse(embed_dim=self.hidden_dim, sparse_ratio=self.sparse_ratio)
        self.aggr_net = TokenAggregation(dim=self.hidden_dim, keeped_patches=self.keeped_patches)
        self.caa = CAA(channels=self.hidden_dim)

    def forward(self, img_embs, cap_embs, cap_lens):
        B_v, L_v, C = img_embs.shape
        img_embs_norm = F.normalize(img_embs, dim=-1)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)
        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True
        
        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]
            img_cls_emb_norm = img_embs_norm[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        with torch.no_grad():
            img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

        improve_sims = []
        score_mask_all = []

        for i in range(len(cap_lens)):
            n_word = cap_lens[i]                  
            cap_i = cap_embs[i, :n_word, :]    
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            # Inside CrossSparseAggrNet_v2's forward method
            with torch.no_grad():

                # Compute global context vectors for both image and text
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
    
                # Ensure cap_i_glo is expanded to match the batch size of img_spatial_embs_norm
                cap_i_glo_expanded = cap_i_glo.expand(B_v, -1, -1)

                # Compute self-attention for image spatial embeddings
                img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
                img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)  # (B_v, L_v)

                # Compute cross-attention between image spatial embeddings and caption global context
                img_spatial_cap_i_attention = (cap_i_glo_expanded * img_spatial_embs_norm).sum(dim=-1)  # (B_v, L_v)

                # Ensure both attention maps have the same shape before adding them together
                if img_spatial_self_attention.shape != img_spatial_cap_i_attention.shape:
                    raise ValueError(f"Shape mismatch: img_spatial_self_attention has shape {img_spatial_self_attention.shape}, "
                         f"but img_spatial_cap_i_attention has shape {img_spatial_cap_i_attention.shape}")

                # Combine the two attention maps
                score = img_spatial_self_attention + img_spatial_cap_i_attention  # (B_v, L_v)

            select_tokens, extra_token, score_mask = self.sparse_net(tokens=img_spatial_embs, 
                                                                    attention_x=img_spatial_self_attention, 
                                                                    attention_y=img_spatial_cap_i_attention.squeeze(1))

            aggr_tokens = self.aggr_net(select_tokens)
            keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)

            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens
            
            select_tokens = F.normalize(select_tokens, dim=-1)
            sim_one_text = mask_xattn_one_text(img_embs=select_tokens, cap_i_expand=cap_i_expand)

            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask)

        improve_sims = torch.cat(improve_sims, dim=1)
        score_mask_all = torch.stack(score_mask_all, dim=0)

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims


if __name__ == '__main__':
    pass