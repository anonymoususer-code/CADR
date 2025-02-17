import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, SwinModel
import logging
from transformers import AutoModel
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def get_text_encoder(opt):
    txt_enc = EncoderText_BERT(opt)   
    return txt_enc


def get_image_encoder(opt):
    img_enc = VisionTransEncoder(opt)
    return img_enc

#新添加的编码模块
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.5):
       super(CrossAttention, self).__init__()
       self.embed_dim = embed_dim
       self.dropout = dropout
       self.modality_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

       self.task_encoder = nn.Sequential(
           nn.Linear(embed_dim, embed_dim * 3),
           nn.ReLU(),
           nn.Linear(embed_dim * 3, embed_dim),
           nn.Sigmoid(),
        )

       self.q_in_proj = nn.Linear(embed_dim, embed_dim)
       self.k_in_proj = nn.Linear(embed_dim, embed_dim)
       self.v_in_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
       u = self.modality_encoder(x)
       r = self.task_encoder(x)

       uq = self.q_in_proj(u)
       uk = self.k_in_proj(u)
       uv = self.v_in_proj(u)

       rq = self.q_in_proj(r)
       rk = self.k_in_proj(r)
       rv = self.v_in_proj(r)

       sigma_u = torch.bmm(uq, rk.transpose(1, 2))
       sigma_u = torch.softmax(sigma_u, dim=2)
       sigma_u = F.dropout(sigma_u, p=self.dropout, training=self.training)
       ru = torch.bmm(sigma_u, rv)

       sigma_r = torch.bmm(rq, uk.transpose(1, 2))
       sigma_r = torch.softmax(sigma_r, dim=2)
       sigma_r = F.dropout(sigma_r, p=self.dropout, training=self.training)
       ur = torch.bmm(sigma_r, uv)

       ru = (ru + ur) / 2.0
       return  0.6*ru


# ViT encoder
class VisionTransEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # Swin model
        if 'swin' in opt.vit_type:                           
            # img_res 224 * 224, 7*7 patch
            self.visual_encoder = SwinModel.from_pretrained("/workspace/dataset/private/dataset/swin")
        
            opt.num_patches = 49
            print('swin model')
            
        if 'dinov2' in opt.vit_type: 
            #img_res 518 * 518, "patch_size": 14
            self.visual_encoder = AutoModel.from_pretrained('/workspace/dataset/private/dataset/dinov2')
            opt.num_patches = 28
            print('DINOv2 model')    

        # dimension transform
        if opt.embed_size == self.visual_encoder.config.hidden_size:
            self.vision_proj = nn.Identity()
        else:
            self.vision_proj = nn.Linear(self.visual_encoder.config.hidden_size, opt.embed_size)            
        
        # 添加交叉注意力层
        self.cross_attn = CrossAttention(opt.embed_size)
    def forward(self, images):
    
        # (B, L_v, C_hidden)
        img_feats = self.visual_encoder(images).last_hidden_state 

        # the dimension transform
        # (B, L_v, C)
        img_feats = self.vision_proj(img_feats)

        img_feats = self.cross_attn(img_feats)+img_feats
        
        return img_feats  
        
    def freeze_backbone(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.visual_encoder.parameters():  
            param.requires_grad = True     


# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt
        self.embed_size = opt.embed_size
        
        self.tokenizer = BertTokenizer.from_pretrained('/workspace/dataset/private/dataset/bert')
        self.bert = BertModel.from_pretrained('/workspace/dataset/private/dataset/bert')
        
        if opt.embed_size == self.bert.config.hidden_size:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.bert.config.hidden_size, opt.embed_size)
        
        # 添加交叉注意力层
        self.cross_attn = CrossAttention(opt.embed_size)
    def forward(self, x, lengths):

        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()

        # all hidden features, D=768 in bert-base model
        # attention_mask： Mask to avoid performing attention on padding token indices.
        # bert_output[0] is the last/final hidden states of all tokens
        # bert_output[1] is the hidden state of [CLS] + one fc layer + Tanh, can be used for classification tasks.

        # N = max_cap_lengths, D = 768
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D

        # B x N x embed_size
        cap_emb = self.fc(bert_emb)
        cap_emb = self.cross_attn(cap_emb)+cap_emb
        
        return cap_emb        

    def freeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.bert.parameters():  
            param.requires_grad = True  


if __name__ == '__main__':

    pass
