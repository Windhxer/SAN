import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import CNNBlockBase, Conv2d
from torch import nn
from torch.nn import functional as F
import einops


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNormBlock(nn.Module):
    def __init__(self, dim, fn, eps=1e-6, dropout=0., pre_norm=True):
        super(LayerNormBlock, self).__init__()
        self.norm = nn.LayerNorm(dim, eps)
        self.dropout = nn.Dropout(dropout)
        self.fn = fn
        self.pre_norm = pre_norm

    def forward(self, x, **kwargs):
        if self.pre_norm:
            return self.dropout(self.fn(self.norm(x), **kwargs))
        else:
            return self.norm(self.dropout(self.fn(x, **kwargs)))


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0, kmax=False):
        super(Attention, self).__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.all_head_size = dim_head * num_heads
        self.scale = dim_head ** -0.5

        self.attn_prob = nn.Softmax(dim=-1)
        self.query = nn.Linear(dim, inner_dim, bias=False)
        self.key = nn.Linear(dim, inner_dim, bias=False)
        self.value = nn.Linear(dim, inner_dim, bias=False)
        self.kmax = kmax
        
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, 
                x,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False):

        h = self.num_heads
        if encoder_hidden_states is not None:
            b = encoder_hidden_states.shape[0]
            if len(x.shape) == 2:
                x = einops.repeat(x.unsqueeze(0), "() n d -> b n d", b=b)
            attention_mask = encoder_attention_mask
            qkv = (self.query(x), self.key(encoder_hidden_states), self.value(encoder_hidden_states))
        else:
            qkv = (self.query(x), self.key(x), self.value(x))

        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if attention_mask is not None:
            dots = dots + attention_mask.to(dots.dtype)
        
        if self.kmax:
            # attn = F.one_hot(torch.argmax(dots, axis=-2), num_classes=x.shape[-2]).transpose(0, 1, 3, 2)
            attn = F.gumbel_softmax(dots, tau=1., dim=-2, hard=False)
        else:
            attn = self.attn_prob(dots)
        if head_mask is not None:
            attn = attn * head_mask

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if output_attentions:
            return (out, attn)
        
        return out


class MeanShiftAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0, kappa=30, kmax=False):
        super(MeanShiftAttention, self).__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.all_head_size = dim_head * num_heads
        # self.scale = dim_head ** -0.5
        self.scale = kappa

        self.attn_prob = nn.Softmax(dim=-1)
        self.query = nn.Linear(dim, inner_dim, bias=False)
        self.key = nn.Linear(dim, inner_dim, bias=False)
        self.value = nn.Linear(dim, inner_dim, bias=False)
        self.kmax = kmax
        
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, 
                x,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False):

        h = self.num_heads
        if encoder_hidden_states is not None:
            b = encoder_hidden_states.shape[0]
            if len(x.shape) == 2:
                x = einops.repeat(x.unsqueeze(0), "() n d -> b n d", b=b)
            attention_mask = encoder_attention_mask
            qkv = (F.normalize(self.query(x), dim=-1), F.normalize(self.key(encoder_hidden_states), dim=-1), self.value(encoder_hidden_states))
        else:
            qkv = (F.normalize(self.query(x), dim=-1), F.normalize(self.key(x), dim=-1), self.value(x))

        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if attention_mask is not None:
            dots = dots + attention_mask.to(dots.dtype)
        
        if self.kmax:
            # attn = F.one_hot(torch.argmax(dots, axis=-2), num_classes=x.shape[-2]).transpose(0, 1, 3, 2)
            attn = F.gumbel_softmax(dots, tau=1., dim=-2, hard=False)
        else:
            attn = self.attn_prob(dots)
        if head_mask is not None:
            attn = attn * head_mask

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = F.normalize(out, dim=-1)
        out = self.to_out(out)
        if output_attentions:
            return (out, attn)
        
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LayerNormBlock(dim, Attention(dim, num_heads=heads, dim_head=dim_head, dropout=dropout)),
                LayerNormBlock(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])
        ] * depth)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class kMaxTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(kMaxTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LayerNormBlock(dim, Attention(dim, num_heads=heads, dim_head=dim_head, dropout=dropout, kmax=True)),
                LayerNormBlock(dim, Attention(dim, num_heads=heads, dim_head=dim_head, dropout=dropout)),
                LayerNormBlock(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])
        ] * depth)

    def forward(self, x, encoder_hidden_states):
        b = encoder_hidden_states.shape[0]
        if len(x.shape) == 2:
            x = einops.repeat(x.unsqueeze(0), "() n d -> b n d", b=b)
            
        for kmax_attn, self_attn, ff in self.layers:
            x = kmax_attn(x, encoder_hidden_states=encoder_hidden_states) + x
            x = self_attn(x) + x
            x = ff(x) + x
        return x


class MeanShiftTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(MeanShiftTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LayerNormBlock(dim, MeanShiftAttention(dim, num_heads=heads, dim_head=dim_head, dropout=dropout)),
                LayerNormBlock(dim, MeanShiftAttention(dim, num_heads=heads, dim_head=dim_head, dropout=dropout)),
                LayerNormBlock(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])
        ] * depth)

    def forward(self, x, encoder_hidden_states):
        b = encoder_hidden_states.shape[0]
        if len(x.shape) == 2:
            x = einops.repeat(x.unsqueeze(0), "() n d -> b n d", b=b)
            
        for ms_attn, self_attn, ff in self.layers:
            x = ms_attn(x, encoder_hidden_states=encoder_hidden_states) + x
            x = self_attn(x) + x
            x = ff(x) + x
            x = F.normalize(x, dim=-1)
        return x


def quantize(z, query_embed, beta=0.25):
    # z = F.normalize(z, dim=-1)
    # query_embed = F.normalize(query_embed, dim=-1)
    dist = torch.cdist(z, query_embed, p=2)
    encoding_index = torch.argmin(dist, dim=-1)
    encoding = torch.zeros(*dist.shape, device=z.device)
    encoding.scatter_(-1, encoding_index.unsqueeze(-1), 1) # 16384, 512
    z_q = torch.matmul(encoding, query_embed).view(z.shape)
    z_q = z + (z_q - z).detach()

    q_latent_loss = F.mse_loss(z_q, z.detach()) 
    e_latent_loss = F.mse_loss(z_q.detach(), z)
    loss_vq = beta * q_latent_loss + e_latent_loss
    return z_q, loss_vq, dist, encoding_index


class SLSTransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, downsample=1.):
        super(SLSTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LayerNormBlock(dim, Attention(dim, num_heads=heads, dim_head=dim_head, dropout=dropout)),
                LayerNormBlock(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])
        ] * depth)
        self.ps = int(32 * downsample)
        self.pn = 20
        self.unpatch_conv = nn.ConvTranspose2d(dim, 3, self.ps, self.ps)
    
    def forward(self, x, quant=True):
        # The input consists of [SLS_TOKENS+CLS_TOKEN+CLIP_VISUAL_TOKENS, bs, dim] = [100+1+20x20, bs, dim]
        # CLIP patch_size = 32, patch_num = 20
        x = x.permute(1, 0, 2)
        if quant:
            sls_tokens = x[:, :-self.pn**2-1]
            cls_token = x[:, -self.pn**2-1:-self.pn**2]
            x = x[:, -self.pn**2:]
            x_q, loss_vq, _, _ = quantize(x, sls_tokens) # here does we need loss vq?
            x_q = torch.cat([cls_token, x_q], dim=1)
        else:
            x_q = x.clone()


        for attn, ff in self.layers:
            x_q = attn(x_q) + x_q
            x_q = ff(x_q) + x_q

        x_vit = x_q[:, -self.pn**2:, :].view(-1, self.pn, self.pn, x_q.shape[-1]).permute(0, 3, 1, 2)
        x_conv = self.unpatch_conv(x_vit)
        return x_vit, x_conv

class AddFusion(CNNBlockBase):
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__(in_channels, out_channels, 1)
        self.input_proj = nn.Sequential(
            LayerNorm(in_channels),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
        )
        weight_init.c2_xavier_fill(self.input_proj[-1])
        self.query_length = cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        # x: [N,L,C] y: [N,C,H,W]
        Q = self.query_length
        query_embed = x[:, :Q, :]
        x = x[:, Q:, :]

        y = (
            F.interpolate(
                self.input_proj(y.contiguous()),
                size=spatial_shape,
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(x.shape)
        )
        x = x + y
        x = torch.cat([query_embed, x], dim=1)
        loss_placeholder = torch.tensor(0.0, requires_grad=False).to(x.device)
        return x, loss_placeholder


class VQFusion(CNNBlockBase):
    def __init__(self, in_channels, out_channels, cfg, beta=0.25):
        super().__init__(in_channels, out_channels, 1)
        self.input_proj = nn.Sequential(
            LayerNorm(in_channels),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
        )
        weight_init.c2_xavier_fill(self.input_proj[-1])
        
        self.cross_attn = MeanShiftTransformer(
            dim=240,
            depth=2,
            heads=4,
            dim_head=120,
            mlp_dim=640,
            dropout=0.,
        )

        assert cfg is not None
        self.query_length = cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES
        self.beta = beta

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        # x: [N,Q+L,C] y: [N,C,H,W], H=W=40
        Q = self.query_length
        query_embed = x[:, :Q, :]
        x = x[:, Q:, :]

        y = (
            F.interpolate(
                self.input_proj(y.contiguous()), 
                size=spatial_shape, 
                mode="bilinear", 
                align_corners=False
            )
            .permute(0, 2, 3, 1)
            .reshape(x.shape)
        )

        # Inject VQ here
        y_q, loss_vq, _, indices = quantize(y, query_embed, beta=self.beta)
        x = x + y_q
        query_embed = self.cross_attn(query_embed, y)
        x = torch.cat([query_embed, x], dim=1)
        return x, loss_vq


def build_fusion_layer(fusion_type: str, in_channels: int, out_channels: int, cfg=None):
    if fusion_type == "add":
        return AddFusion(in_channels, out_channels, cfg)
    elif fusion_type == "vq":
        return VQFusion(in_channels, out_channels, cfg)
    else:
        raise ValueError("Unknown fusion type: {}".format(fusion_type))
