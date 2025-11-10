"""
This file is based on the code from: https://openreview.net/forum?id=BllUWdpIOA
"""

import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from copy import deepcopy
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio
from utils.misc import ema_update_model


@torch.no_grad()
def kernel(
    model,
    src_model,
    bias=0.99,
    normalization_constant=1e-4
):
    energy_buffer = []
    for param, src_param in zip(model.parameters(), src_model.parameters()):
        energy = F.cosine_similarity(
            src_param.data.flatten(),
            param.flatten(),
            dim=-1)

        energy_buffer.append(energy)

    energy = torch.stack(energy_buffer, dim=0).mean()
    energy = (bias - energy) / normalization_constant

    return energy


@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x


@ADAPTATION_REGISTRY.register()
class CMF(TTAMethod):
    def __init__(self, cfg, model, combiner, num_classes):
        super().__init__(cfg, model, combiner, num_classes)
        param_size_ratio = self.num_trainable_params / 38400

        self.cfg = cfg
        self.use_weighting = cfg.ROID.USE_WEIGHTING
        self.use_prior_correction = cfg.ROID.USE_PRIOR_CORRECTION
        self.use_consistency = cfg.ROID.USE_CONSISTENCY
        self.momentum_probs = cfg.ROID.MOMENTUM_PROBS
        self.temperature = cfg.ROID.TEMPERATURE
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).cuda()
        self.tta_transform = get_tta_transforms(self.img_size, padding_mode="reflect", cotta_augs=False)

        # setup loss functions
        self.sce = SymmetricCrossEntropy()
        self.slr = SoftLikelihoodRatio()
        self.ent = Entropy()

        # copy and freeze the source model


        self.src_model = deepcopy(self.model)
        for param in self.src_model.parameters():
            param.detach_()

        # CMF
        self.alpha = cfg.CMF.ALPHA
        self.gamma = cfg.CMF.GAMMA
        self.post_type = cfg.CMF.TYPE
        self.hidden_model = deepcopy(self.model)
        for param in self.hidden_model.parameters():
            param.detach_()

        self.hidden_var = 0
        self.q = cfg.CMF.Q * param_size_ratio

        self.models = [self.src_model, self.model, self.hidden_model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        if self.cfg.MODEL.USE_CLIP:
            self.logit_scale = self.model.clip_model.logit_scale.exp()
        else:
            self.logit_scale = self.model.logit_scale.exp()

    @torch.no_grad()
    def bayesian_filtering(self):
        # 1. predict step
        # NOTE: self.post_type==lp is the default,
        # in which case the predict step and update step can be combined to reduce computation.
        # For clarity, they are separated in the code.
        recovered_model = ema_update_model(
            model_to_update=self.hidden_model,
            model_to_merge=self.src_model,
            momentum=self.alpha,
            device=self.device,
            update_all=True
        )

        # 2. update step
        self.hidden_var = self.alpha ** 2 * self.hidden_var + self.q

        r = (1 - self.q)
        self.beta = r / (self.hidden_var + r)
        self.beta = self.beta if self.beta > 0.89 else 0.89
        self.beta = self.beta if self.beta < 0.9999 else 1.0

        self.hidden_var = self.beta * self.hidden_var
        self.hidden_model = ema_update_model(
            model_to_update=recovered_model,
            model_to_merge=self.model,
            momentum=self.beta,
            device=self.device,
            update_all=True
        )

        # 3. parameter ensemble step
        self.model = ema_update_model(
            model_to_update=self.model,
            model_to_merge=recovered_model if self.post_type == "op" else self.hidden_model,
            momentum=self.gamma,
            device=self.device
        )

        # logging
        if self.cfg.TEST.DEBUG:
            tgt_energy = kernel(
                model=self.model,
                src_model=self.src_model,
                bias=0,
                normalization_constant=1.0
            )
            hidden_energy = kernel(
                model=self.hidden_model,
                src_model=self.src_model,
                bias=0,
                normalization_constant=1.0
            )
            res ={
                "tgt_energy": tgt_energy,
                "hidden_energy": hidden_energy,
            }
        else:
            res = None

        return res

    def loss_calculation(self, query_image, query_text, target_images_pool):
       
        query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool, need_target=True)

        outputs = self.logit_scale * lerp_features @ target_image_features.t()

        if self.use_weighting:
            with torch.no_grad():
                # calculate diversity based weight
                # print(self.class_probs_ema.unsqueeze(dim=0).shape)  # [1, bs64]
                # print(outputs.softmax(1).shape)  # [bs64, cls(target_image_features)]
                if self.class_probs_ema.shape[0] != outputs.shape[1]:
                    self.num_classes = outputs.shape[1]
                    self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(outputs.device)
                
                weights_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1)
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()

                # calculate certainty based weight
                weights_cert = - self.ent(logits=outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())

                # calculate the final weights
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.

                self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs.softmax(1).mean(0), momentum=self.momentum_probs)

        # calculate the soft likelihood ratio loss
        loss_out = self.slr(logits=outputs)

        # weight the loss
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / self.batch_size

        # calculate the consistency loss
        if self.use_consistency:
            x1 = self.tta_transform(query_image[~mask])
            if self.cfg.MODEL.USE_CLIP:
                query_text = self.model.tokenize(query_text).to(self.model.device)
                y1 = query_text[~mask]
            else:
                y1 = query_text
            
            z1 = target_images_pool

            con_query_image_features = self.model.get_image_features(x1, Global_only=True)

            con_target_image_features = z1

            con_caption_features = self.model.get_text_features(tokenized_prompts=y1)

            con_mixture_features = 0.5 * con_query_image_features + 0.5 * con_caption_features 
            con_mixture_features = con_mixture_features / con_mixture_features.norm(dim=-1, keepdim=True)

            con_lerp_features = (1-self.cfg.OPTIM.MIXTURE_FACTOR_TEXT) * con_query_image_features + self.cfg.OPTIM.MIXTURE_FACTOR_TEXT * con_caption_features
            con_lerp_features = con_lerp_features / con_lerp_features.norm(dim=-1, keepdim=True)

            outputs_aug = self.logit_scale * con_lerp_features @ con_target_image_features.t()

            loss += (self.sce(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / self.batch_size

            return query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, query_image, query_text, target_images_pool):
        for _ in range(self.steps):
            if query_image.shape[0] != 64:
                self.num_classes = query_image.shape[0]
                self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(self.device)

            if self.mixed_precision and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, outputs, loss = self.loss_calculation(query_image, query_text, target_images_pool)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, outputs, loss = self.loss_calculation(query_image, query_text, target_images_pool)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()    

            with torch.no_grad():
                self.bayesian_filtering()

                if self.use_prior_correction:
                    prior = outputs.softmax(1).mean(0)
                    smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                    smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                    outputs *= smoothed_prior

            with torch.no_grad():
                query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool)

        return query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(self.device)

    def collect_params(self, fixed_v_lay_list=None, fixed_t_lay_list=None):
        """Collect the affine scale + shift parameters from normalization layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        if (fixed_v_lay_list is None or len(fixed_v_lay_list) == 0) and (fixed_t_lay_list is None or len(fixed_t_lay_list) == 0):
            for nm, m in self.model.named_modules():
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
            return params, names
        else:
            if fixed_v_lay_list is not None:
                params_v = []
                names_v = []
                for nm, m in self.model.named_modules():
                    for np, p in m.named_parameters():
                        if (np in ['weight', 'bias']
                                and p.requires_grad
                                and not any(f"{nm}.{np}".startswith(f"{prefix}") for prefix in fixed_v_lay_list)
                                and "visual" in f"{nm}.{np}"):
                            params_v.append(p)
                            names_v.append(f"{nm}.{np}")
                params = params + params_v
                names = names + names_v
            if fixed_t_lay_list is not None:
                params_t = []
                names_t = []
                for nm, m in self.model.named_modules():
                    for np, p in m.named_parameters():
                        if (np in ['weight', 'bias']
                                and p.requires_grad
                                and not any(f"{nm}.{np}".startswith(f"{prefix}") for prefix in fixed_t_lay_list)
                                and "visual" not in f"{nm}.{np}"):
                            params_t.append(p)
                            names_t.append(f"{nm}.{np}")
                params = params + params_t
                names = names + names_t

            return params, names

        return params, names

    def configure_model(self):
        """Configure model."""
        self.model.eval()
        self.model.requires_grad_(False)
        # re-enable gradient for normalization layers
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
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
