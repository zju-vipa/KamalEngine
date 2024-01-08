from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])    # 14*14
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        B, C, H, W = x.shape   # B:batch_size, C:channel, H:height, W:width
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)   # batch_size, 196, 768
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个head的dimension
        self.scale = qk_scale or head_dim ** -0.5  # 根号d分之一
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # Wo
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # x.shape: [batch_size, num_patches + 1, total_embed_dim(768)]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v切片后形状：[batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]   最后两维换位置
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]     矩阵乘法，最后两维做
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  #每行进行softmax处理
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim] (每个head concat起来)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,  # 第一个全连接层的节点个数是输入节点个数的4倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0,
                 attn_drop_ratio=0,
                 drop_path_ratio=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 乘4
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,   # input image size
                 patch_size=16,  # patch size
                 in_c=3,         # number of input channels
                 num_classes=1000,  # number of classes for classification head
                 embed_dim=768,     # embedding dimension
                 depth=12,          # encoder重复的次数
                 num_heads=12,      # number of attention heads
                 mlp_ratio=4.0,     # ratio of mlp hidden dim to embedding dim
                 qkv_bias=True,     # enable bias for qkv if True
                 qk_scale=None,     # override default qk scale of head_dim ** -0.5 if set
                 representation_size=None,  # 有无pre_logits
                 distilled=False,   # model includes a distillation token and head as in DeiT models
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None,
                 is_student=False):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.is_student = is_student
        if self.is_student:
            self.num_patches = self.patch_embed.num_patches * 2
        else:
            self.num_patches = self.patch_embed.num_patches

        self.FC = nn.Linear(self.patch_embed.num_patches, self.patch_embed.num_patches * 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))  # 197*768 or 198*768
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule, 每一个encoder block里面的drop_path_ratio是递增的
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])  # 重复L次的encoder block
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)


    def forward_features(self, x):
        # x: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        token = x
        if self.is_student:
            x = x.transpose(1, 2)
            x = self.FC(x)
            x = x.transpose(1, 2)   # [B, num_patches * 2, embed_dim]
            token = x
        #########################################################################################################
        # cls_token: [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:     # 拼接
            x = torch.cat((cls_token, x), dim=1)  # [B, 196*2+1, 768] (if student)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)  # 位置编码，然后通过dropout层
        encoder_results = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            encoder_results.append(x[:, 1:, :])    # 去掉第一个cls token
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), token, encoder_results
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x, token, encoder_results = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, token, encoder_results


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000, is_student=False, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.):
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None, num_classes=num_classes, is_student=is_student,
                              drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)
    return model


def vit_large_patch16_224(num_classes: int = 1000, is_student=False):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes,
                              is_student=is_student)
    return model
