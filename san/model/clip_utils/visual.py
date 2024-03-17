from typing import List
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from open_clip.transformer import VisionTransformer
from detectron2.layers import ShapeSpec
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from ..perceptual import LPIPS
from ..attn_helper import cross_attn_layer, downsample2d, resize_pos_embed2d


def get_last_used_number(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter out non-numeric filenames
    numbers = [int(file.split('.')[0]) for file in files if file.split('.')[0].isdigit()]

    # Return the maximum number or 0 if no files exist
    return max(numbers, default=0)


class ClipOutput(dict):
    def __init__(self, spacial_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacial_shape = spacial_shape

    def save(self, idx: int, clip_feat: torch.Tensor):
        l, n, c = clip_feat.shape
        self[idx] = (
            clip_feat[1:].permute(1, 2, 0).reshape(n, c, *self.spacial_shape)
        )  # n, c, h, w
        self[f"{idx}_cls_token"] = clip_feat[0:1]  # 1, n, c


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        visual_encoder: VisionTransformer,
        last_layer_idx: int = -1,
        frozen_exclude=[],
    ):
        super().__init__()
        self.output_tokens = visual_encoder.output_tokens
        self.image_size = visual_encoder.image_size
        self.patch_size = visual_encoder.patch_size
        self.grid_size = visual_encoder.grid_size
        self.num_features = visual_encoder.ln_pre.normalized_shape[0]

        self.input_patchnorm = visual_encoder.input_patchnorm
        self.patchnorm_pre_ln = visual_encoder.patchnorm_pre_ln
        self.conv1 = visual_encoder.conv1

        # class embeddings and positional embeddings
        self.class_embedding = visual_encoder.class_embedding
        self.positional_embedding = visual_encoder.positional_embedding
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = visual_encoder.patch_dropout
        self.ln_pre = visual_encoder.ln_pre
        if last_layer_idx == -1:
            self.resblocks = visual_encoder.transformer.resblocks
            self.last_output_idx = len(self.resblocks) + 1
        else:
            self.resblocks = visual_encoder.transformer.resblocks[:last_layer_idx]
            self.last_output_idx = last_layer_idx + 1
        #
        self.frozen_exclude = frozen_exclude
        self._freeze(self.frozen_exclude)

    def forward(self, x: torch.Tensor):
        if self.input_patchnorm:
            raise NotImplementedError("input_patchnorm is not implemented yet.")
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            _, _, h, w = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        pos_embed = self.positional_embedding.to(x.dtype)
        pos_embed = resize_pos_embed2d(pos_embed[None, ...], self.grid_size, (h, w))[0]
        x = x + pos_embed

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = ClipOutput(spacial_shape=(h, w))
        outputs.save(0, x)
        for i, resblock in enumerate(self.resblocks, start=1):
            x = resblock(x)
            outputs.save(i, x)
        return outputs

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False

    @property
    def output_shapes(self):
        return {
            i: ShapeSpec(channels=self.num_features)
            for i in range(self.last_output_idx)
        }

    @property
    def size_divisibility(self):
        return self.patch_size[0]


class RecWithAttnbiasHead(nn.Module):
    def __init__(
        self,
        visual_encoder: VisionTransformer,
        visual_decoder: VisionTransformer,
        log_dir: str = "output/debug",
        first_layer_idx: int = 0,
        frozen_exclude: List[str] = [],
        SLS_TOKEN_FORMAT: str = "cls_token",
        sls_token_num: int = 1,
        cross_attn: bool = True,
        downsample_method: str = "bilinear",
    ):
        super().__init__()
        # determine save_fig_name
        self.log_dir = log_dir + '/reconstructions/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.output_tokens = visual_encoder.output_tokens
        self.output_dim = visual_encoder.output_dim
        self.first_layer_idx = first_layer_idx
        self.cross_attn = cross_attn
        self.downsample_method = downsample_method

        if first_layer_idx < 0:
            raise NotImplementedError("first_layer_idx < 0 is not implemented yet.")
        self.resblocks = visual_encoder.transformer.resblocks[first_layer_idx:]
        self.global_average_pool = visual_encoder.global_average_pool
        self.attn_pool = visual_encoder.attn_pool
        assert (
            self.attn_pool is None
        ), "recognition with attn_pool is not implemented yet."
        assert (
            not self.global_average_pool
        ), "recognition with global_average_pool is not implemented yet."
        self.ln_post = visual_encoder.ln_post
        self.proj = visual_encoder.proj

        self.SLS_TOKEN_FORMAT = SLS_TOKEN_FORMAT
        self.sls_token_num = sls_token_num
        self.frozen_exclude = frozen_exclude

        if SLS_TOKEN_FORMAT in ["learnable_token", "pos_embedding"]:
            self.sls_token = nn.Parameter(
                torch.randn(sls_token_num, 1, self.proj.shape[0])
            )
            nn.init.normal_(self.sls_token, std=0.02)
            self.frozen_exclude.append("sls_token")
        self._freeze(self.frozen_exclude)

        # Add trainable reconstruction modules
        self.save_num = get_last_used_number(self.log_dir) + 1
        self.save_per = 1000
        self.decoder = visual_decoder
        self.decoder_act = nn.Tanh()
        # self.lpips = LPIPS(net_type='vgg').eval()
        self.lpips = LPIPS().eval()

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False

    def forward(self, features, attn_bias, query_token, normalize: bool = False):
        # construct clip shadow features.
        cls_token = features[f"{self.first_layer_idx}_cls_token"]  # 1,n,c
        pix_feat = features[self.first_layer_idx]  # n,c,h,w
        n, c, h, w = pix_feat.shape
        x = torch.cat(
            [cls_token, pix_feat.reshape(n, c, -1).permute(2, 0, 1)]
        )  # 1+l,n,c

        # construct sls token.
        if self.SLS_TOKEN_FORMAT == "cls_token":
            sls_token = cls_token.repeat(self.sls_token_num, 1, 1)
        elif self.SLS_TOKEN_FORMAT == "learnable_token":
            sls_token = self.sls_token.expand(-1, n, -1)
        elif self.SLS_TOKEN_FORMAT == "pos_embedding":
            sls_token = self.sls_token.expand(-1, n, -1) + cls_token
        elif self.SLS_TOKEN_FORMAT == "query_token":
            sls_token = query_token
        
        # sls_token = query_token + sls_token

        # construct attn biases.
        attn_biases = self._build_attn_biases(attn_bias, target_shape=(h, w))
        if self.cross_attn:
            for i, resblock in enumerate(self.resblocks):
                if self.cross_attn:
                    sls_token = cross_attn_layer(
                        resblock,
                        sls_token,
                        x[1:,],
                        attn_biases[i],
                    )
                    if i < len(self.resblocks) - 1:
                        x = resblock(x)
        else:
            x = torch.cat([sls_token, x], dim=0)
            bs = x.shape[1]
            log_idx = torch.randint(high=bs, size=(1,))[0]
            loss_mse = []

            # log reconstruction figures
            if self.save_num % self.save_per == 0:
                fig = plt.figure(figsize=(16 + len(self.resblocks) * 16, 20))
                ax = fig.add_subplot(1, 1 + len(self.resblocks), 1)
                ax.imshow((features['input'][log_idx] * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy(), interpolation='none')
                ax.set_axis_off()

            for i, resblock in enumerate(self.resblocks):
                x = resblock(x, attn_mask=attn_biases[i])
                
                # Fused Feature Block, Add multistage manipulations: Reconstruction/Lexiconization
                x_vit, x_recon = self.decoder[i](x, quant=True)

                # determine loss to use: LPIPS, L2, VQGAN?
                if x_vit.shape == pix_feat.shape:
                    # x_recon = self.decoder_act(x_recon)
                    loss_mse_layer = F.mse_loss(x_vit, pix_feat) + 0.1 * (x_vit - pix_feat).abs().mean()
                    x_recon_copy = x_recon.clone().detach().view(x_recon.shape[0], x_recon.shape[1], -1)
                    x_recon_min, x_recon_max = x_recon_copy.aminmax(dim=-1)
                    x_recon_min = x_recon_min.unsqueeze(-1).unsqueeze(-1)
                    x_recon_max = x_recon_max.unsqueeze(-1).unsqueeze(-1) - x_recon_min

                    x_recon = x_recon - x_recon_min #.repeat(1, 1, x_recon.shape[-2], x_recon.shape[-1])
                    if x_recon_max.min() > 0:
                        x_recon = x_recon * 2. / x_recon_max - 1.
                    
                    # loss_mse_layer += self.lpips(x_recon, features['input'])
                    # loss_mse_layer = F.mse_loss(features['input'], x_recon) + 0.1 * (features['input'] - x_recon).abs().mean()
                    loss_mse_layer += self.lpips(features['input'].contiguous(), x_recon.contiguous()).mean()
                    loss_mse.append(loss_mse_layer)

                    if self.save_num % self.save_per == 0:
                        axi = fig.add_subplot(1, 1 + len(self.resblocks), 2 + i)
                        x_display = x_recon[log_idx].clone()
                        x_display = (x_display.detach().float() * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()
                        axi.imshow(x_display, interpolation='none')
                        axi.set_axis_off()

            if self.save_num % self.save_per == 0:
                fig.tight_layout()
                fig.savefig(self.log_dir + f'{self.save_num}.png')
                plt.close(fig)
            self.save_num += 1
            sls_token = x[:self.sls_token_num]

        sls_token = sls_token.permute(1, 0, 2)  # LND -> NLD

        sls_token = self.ln_post(sls_token)

        if self.proj is not None:
            sls_token = sls_token @ self.proj
        if normalize:
            sls_token = F.normalize(sls_token, dim=-1)
        return sls_token, loss_mse

    def _build_attn_biases(self, attn_biases, target_shape):
        formatted_attn_biases = []
        for attn_bias in attn_biases:
            # convert it to proper format: N*num_head,L,L
            # attn_bias: [N, num_head/1, num_sls,H,W]
            n, num_head, num_sls, h, w = attn_bias.shape
            # reshape and downsample
            attn_bias = downsample2d(
                attn_bias.reshape(n, num_head * num_sls, h, w),
                target_shape,
                method=self.downsample_method,
            )
            attn_bias = attn_bias.reshape(n, num_head, num_sls, *target_shape)
            true_num_head = self.resblocks[0].attn.num_heads
            assert (
                num_head == 1 or num_head == true_num_head
            ), f"num_head={num_head} is not supported."
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(n * true_num_head, num_sls, -1)
            L = attn_bias.shape[-1]
            if self.cross_attn:
                # [n*num_head, num_sls, L]
                formatted_attn_biases.append(attn_bias)
            else:
                # [n*num_head, num_sls+1+L, num_sls+1+L]
                new_attn_bias = attn_bias.new_zeros(num_sls + 1 + L, num_sls + 1 + L)
                new_attn_bias[:, :num_sls] = -100
                new_attn_bias[torch.arange(num_sls), torch.arange(num_sls)] = 0
                new_attn_bias[:num_sls, num_sls] = -100
                new_attn_bias = (
                    new_attn_bias[None, ...].expand(n * true_num_head, -1, -1).clone()
                )
                new_attn_bias[..., :num_sls, -L:] = attn_bias
                formatted_attn_biases.append(new_attn_bias)

        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [formatted_attn_biases[0] for _ in self.resblocks]
        return formatted_attn_biases
