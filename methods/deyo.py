"""
Builds upon: https://github.com/Jhyun17/DeYO/blob/main/methods/deyo.py
Corresponding paper: https://openreview.net/pdf?id=9w3iw8wDuE
"""

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
from einops import rearrange

from methods.base import TTAMethod
from utils.losses import Entropy
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class DeYO(TTAMethod):
    def __init__(self, cfg, model, combiner, num_classes):
        super().__init__(cfg, model, combiner, num_classes)

        self.cfg = cfg

        self.reweight_ent = cfg.DEYO.REWEIGHT_ENT
        self.reweight_plpd = cfg.DEYO.REWEIGHT_PLPD

        self.plpd_threshold = cfg.DEYO.PLPD
        self.deyo_margin = cfg.DEYO.MARGIN * math.log(num_classes)
        self.margin_e0 = cfg.EATA.MARGIN_E0 * math.log(num_classes)

        self.aug_type = cfg.DEYO.AUG_TYPE
        self.occlusion_size = cfg.DEYO.OCCLUSION_SIZE
        self.row_start = cfg.DEYO.ROW_START
        self.column_start = cfg.DEYO.COLUMN_START
        self.patch_len = cfg.DEYO.PATCH_LEN

        if self.cfg.MODEL.USE_CLIP:
            self.logit_scale = self.model.clip_model.logit_scale.exp()
        else:
            self.logit_scale = self.model.logit_scale.exp()

        self.ent = Entropy()

    def loss_calculation(self, query_image, query_text, target_images_pool):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        
        query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool, need_target=True)


        outputs = self.logit_scale * lerp_features @ target_image_features.t()

        entropys = self.ent(outputs)
        filter_ids_1 = torch.where((entropys < self.deyo_margin))
        entropys = entropys[filter_ids_1]
        if len(entropys) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, outputs, loss

        query_text = self.model.tokenize(query_text).to(self.model.device)
        x_prime = query_image[filter_ids_1]
        x_prime = x_prime.detach()

        y_prime = query_text[filter_ids_1]
        y_prime = y_prime.detach()

        z_prime = target_images_pool
        z_prime = z_prime.detach()

        if self.aug_type == 'occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
            x_prime[:, :, self.row_start:self.row_start + self.occlusion_size, self.column_start:self.column_start + self.occlusion_size] = occlusion_window
        elif self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(((query_image.shape[-1] // self.patch_len) * self.patch_len, (query_image.shape[-1] // self.patch_len) * self.patch_len))
            resize_o = torchvision.transforms.Resize((query_image.shape[-1], query_image.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.patch_len, ps2=self.patch_len)
            x_prime = resize_o(x_prime)
        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])

        with torch.no_grad():
            prime_query_image_features = self.model.get_image_features(x_prime, Global_only=True)

            prime_target_image_features = z_prime

            prime_caption_features = self.model.get_text_features(tokenized_prompts=y_prime)

            prime_mixture_features = 0.5 * prime_query_image_features + 0.5 * prime_caption_features
            prime_mixture_features = prime_mixture_features / prime_mixture_features.norm(dim=-1, keepdim=True)

            prime_lerp_features = (1-self.cfg.OPTIM.MIXTURE_FACTOR_TEXT) * prime_query_image_features + self.cfg.OPTIM.MIXTURE_FACTOR_TEXT * prime_caption_features
            prime_lerp_features = prime_lerp_features / prime_lerp_features.norm(dim=-1, keepdim=True)

            outputs_prime = self.logit_scale * prime_lerp_features @ prime_target_image_features.t()

        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim=1)

        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
        plpd = plpd.reshape(-1)

        filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        entropys = entropys[filter_ids_2]
        if len(entropys) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, outputs, loss

        plpd = plpd[filter_ids_2]

        if self.reweight_ent or self.reweight_plpd:
            coeff = (float(self.reweight_ent) * (1. / (torch.exp(((entropys.clone().detach()) - self.margin_e0)))) +
                     float(self.reweight_plpd) * (1. / (torch.exp(-1. * plpd.clone().detach())))
                     )
            entropys = entropys.mul(coeff)

        loss = entropys.mean(0)
        return query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, query_image, query_text, target_images_pool):
        for _ in range(self.steps):
            if self.mixed_precision and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, outputs, loss = self.loss_calculation(query_image, query_text, target_images_pool)
                # update model only if not all instances have been filtered
                if loss is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.optimizer.zero_grad()
            else:
                query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, outputs, loss = self.loss_calculation(query_image, query_text, target_images_pool)
                # update model only if not all instances have been filtered
                if loss is not None:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()

            with torch.no_grad():
                query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool)

        return query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features

    def collect_params(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names
    
    def configure_model(self):
        """Configure model for use with DeYO."""
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what DeYO updates
        self.model.requires_grad_(False)
        # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)


    @torch.no_grad()
    def encode_text(self, text):
        return self.model.get_text_features(text)

    @torch.no_grad()
    def encode_image(self, image, Global_only=True):
        return self.model.get_image_features(image, Global_only=Global_only)

    def set_image_features(self, images=None, image_features=None, Global_only=True):
        if images is not None:
            return self.model.set_image_features(images=images, Global_only=Global_only)
        else:
            return self.model.set_image_features(image_features=image_features, Global_only=Global_only)

    def set_text_features(self, text=None, tokenized_prompts=None, text_features=None):
        if text is not None or tokenized_prompts is not None:
            return self.model.set_text_features(text=text, tokenized_prompts=tokenized_prompts)
        else:
            assert text_features is not None
            return self.model.set_text_features(text_features=text_features)

    def encode_mixture(self, text, image):
        return self.model.get_mixture_features(text, image)
