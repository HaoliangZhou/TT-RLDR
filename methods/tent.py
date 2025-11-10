"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""
import torch
import torch.nn as nn
import re
from methods.base import TTAMethod
from numpy.core.defchararray import startswith
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from utils import lr_decay as lrd


@ADAPTATION_REGISTRY.register()
class Tent(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, combiner, num_classes):
        super().__init__(cfg, model, combiner, num_classes)

        # setup loss function
        self.softmax_entropy = Entropy()
        self.cfg = cfg

        if self.cfg.MODEL.USE_CLIP:
            self.logit_scale = self.model.clip_model.logit_scale.exp()
        else:
            self.logit_scale = self.model.logit_scale.exp()

    def loss_calculation(self, query_features, target_features):
        logits_per_query = self.logit_scale * query_features @ target_features.t()

        loss = self.softmax_entropy(logits_per_query).mean(0)
        return loss


    @torch.enable_grad()
    def forward_and_adapt(self, query_image, query_text, target_images_pool):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        
        query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool, need_target=True)

        for i in range(self.steps):
            if self.mixed_precision and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    loss = self.loss_calculation(lerp_features, target_image_features)
                loss.requires_grad_(True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            else:
                loss = self.loss_calculation(lerp_features, target_image_features)
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        with torch.no_grad():
            query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool)


        return query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features



    def collect_params(self, fixed_v_lay_list=None, fixed_t_lay_list=None):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
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
                        if np in ['weight', 'bias'] and p.requires_grad and not any(f"{nm}.{np}".startswith(f"{prefix}") for prefix in fixed_v_lay_list) and "visual" in f"{nm}.{np}":
                            params_v.append(p)
                            names_v.append(f"{nm}.{np}")
                params = params + params_v
                names = names + names_v
            if fixed_t_lay_list is not None:
                params_t = []
                names_t = []
                for nm, m in self.model.named_modules():
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias'] and p.requires_grad and not any(f"{nm}.{np}".startswith(f"{prefix}") for prefix in fixed_t_lay_list)  and "visual" not in f"{nm}.{np}":
                            params_t.append(p)
                            names_t.append(f"{nm}.{np}")
                params = params + params_t
                names = names + names_t

            return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
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

