"""
PromptIR Simplified Model Implementation
"""
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass"""
        weight = self.avg_pool(x)
        weight = self.fc(weight)
        return x * weight


class CrossScaleAttention(nn.Module):
    """Cross-scale attention module"""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q_conv_dec = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_conv_enc = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_conv_enc = nn.Conv2d(dim, dim, kernel_size=1)

        self.q_conv_enc = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_conv_dec = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_conv_dec = nn.Conv2d(dim, dim, kernel_size=1)

        self.out_proj_dec = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj_enc = nn.Conv2d(dim, dim, kernel_size=1)

        self.se_dec = SELayer(dim, reduction=16)
        self.se_enc = SELayer(dim, reduction=16)

    def forward(self, decoder_feat, encoder_feat):
        """Forward pass with cross-scale attention"""
        # pylint: disable=too-many-locals
        _, _, H, W = decoder_feat.shape

        q_dec = self.q_conv_dec(decoder_feat)
        k_enc = self.k_conv_enc(encoder_feat)
        v_enc = self.v_conv_enc(encoder_feat)

        q_enc = self.q_conv_enc(encoder_feat)
        k_dec = self.k_conv_dec(decoder_feat)
        v_dec = self.v_conv_dec(decoder_feat)

        def reshape(x):
            return rearrange(x, 'b (h c) h_ w_ -> b h c (h_ w_)', h=self.heads)

        q_dec = F.normalize(reshape(q_dec), dim=-1)
        k_enc = F.normalize(reshape(k_enc), dim=-1)
        q_enc = F.normalize(reshape(q_enc), dim=-1)
        k_dec = F.normalize(reshape(k_dec), dim=-1)

        attn_dec = torch.matmul(q_dec, k_enc.transpose(-2, -1)) * self.scale
        attn_enc = torch.matmul(q_enc, k_dec.transpose(-2, -1)) * self.scale

        attn_dec = attn_dec.softmax(dim=-1)
        attn_enc = attn_enc.softmax(dim=-1)

        out_dec = torch.matmul(attn_dec, reshape(v_enc))
        out_enc = torch.matmul(attn_enc, reshape(v_dec))

        out_dec = rearrange(out_dec, 'b h c (h_ w_) -> b (h c) h_ w_',
                            h_=H, w_=W)
        out_enc = rearrange(out_enc, 'b h c (h_ w_) -> b (h c) h_ w_',
                            h_=H, w_=W)

        out_dec = self.se_dec(self.out_proj_dec(out_dec))
        out_enc = self.se_enc(self.out_proj_enc(out_enc))

        return decoder_feat + out_dec, encoder_feat + out_enc


def to_3d(x):
    """Convert 4D tensor to 3D"""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Convert 3D tensor to 4D"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """Layer normalization without bias"""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Forward pass"""
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """Layer normalization with bias"""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Forward pass"""
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Layer normalization wrapper"""

    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """Forward pass"""
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Feed forward network"""

    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1,
                                    bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2,
                                kernel_size=3, stride=1, padding=1,
                                groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1,
                                     bias=bias)

    def forward(self, x):
        """Forward pass"""
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3,
                                    stride=1, padding=1, groups=dim*3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """Forward pass"""
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Downsample(nn.Module):
    """Downsampling module"""

    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        """Forward pass"""
        return self.body(x)


class Upsample(nn.Module):
    """Upsampling module"""

    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        """Forward pass"""
        return self.body(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""
    # pylint: disable=too-many-arguments

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        """Forward pass"""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding"""

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1,
                              padding=1, bias=bias)

    def forward(self, x):
        """Forward pass"""
        x = self.proj(x)
        return x


class PromptIR_Simplified(nn.Module):
    """Simplified PromptIR model"""
    # pylint: disable=too-many-instance-attributes,too-many-arguments

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=None,
                 num_refinement_blocks=2,
                 heads=None,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.dim1 = dim
        self.dim2 = int(dim*2)
        self.dim3 = int(dim*2*2)
        self.dim4 = int(dim*2*2*2)

        self.patch_embed = OverlapPatchEmbed(inp_channels, self.dim1)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=self.dim1, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(self.dim1)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=self.dim2, num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(self.dim2)

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=self.dim3, num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(self.dim3)

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=self.dim4, num_heads=heads[3],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(self.dim4)

        self.reduce_chan_level3 = nn.Conv2d(
            self.dim4//2 + self.dim3, self.dim3, kernel_size=1, bias=bias
        )

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=self.dim3, num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(self.dim3)

        self.reduce_chan_level2 = nn.Conv2d(
            self.dim3//2 + self.dim2, self.dim2, kernel_size=1, bias=bias
        )

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=self.dim2, num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(self.dim2)

        self.reduce_chan_level1 = nn.Conv2d(
            self.dim2//2 + self.dim1, self.dim1, kernel_size=1, bias=bias
        )

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=self.dim1, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=self.dim1, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(self.dim1, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=bias)
        self.csa_level3 = CrossScaleAttention(self.dim3)
        self.csa_level2 = CrossScaleAttention(self.dim2)

    def check_image_size(self, x):
        """Check and pad image size to be divisible by 32"""
        _, _, h, w = x.size()
        mod_pad_h = (32 - h % 32) % 32
        mod_pad_w = (32 - w % 32) % 32
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        """Forward pass"""
        inp_img = self.check_image_size(x)

        x = self.patch_embed(inp_img)

        enc1 = self.encoder_level1(x)

        x = self.down1_2(enc1)

        enc2 = self.encoder_level2(x)

        x = self.down2_3(enc2)

        enc3 = self.encoder_level3(x)

        x = self.down3_4(enc3)

        x = self.latent(x)

        x = self.up4_3(x)

        x = torch.cat([x, enc3], 1)
        x = self.reduce_chan_level3(x)

        x = self.decoder_level3(x)

        x = self.up3_2(x)

        x = torch.cat([x, enc2], 1)
        x = self.reduce_chan_level2(x)

        x = self.decoder_level2(x)

        x = self.up2_1(x)

        x = torch.cat([x, enc1], 1)
        x = self.reduce_chan_level1(x)

        x = self.decoder_level1(x)

        x = self.refinement(x)

        x = self.output(x)

        x = x + inp_img

        return x
