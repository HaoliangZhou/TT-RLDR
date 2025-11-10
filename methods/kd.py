"""
Builds upon: https://github.com/oripress/CCC/blob/main/models/rpl.py
Corresponding paper: https://arxiv.org/pdf/2104.12928.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.base import TTAMethod_with_reward
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import GeneralizedCrossEntropy


def kd_distill_loss_v2(logits_student, logits_teacher, T_stu=1.0, T_tea=1.0):
    """
    vanilla KD, KLDiv between teacher and student, only the gradient related part
    """
    log_pred_student = F.log_softmax(logits_student / T_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_tea, dim=1)
    # kl_div = -p log q
    loss_kd = - torch.sum(pred_teacher * log_pred_student, dim=1).mean()
    loss_kd = loss_kd * T_stu * T_stu

    return loss_kd


@ADAPTATION_REGISTRY.register()
class KD(TTAMethod_with_reward):
    """
    KD model for test-time adaptation.
    """
    def __init__(self, cfg, model, reward_model, combiner, num_classes):
        super().__init__(cfg, model, reward_model, combiner, num_classes)

        self.cfg = cfg


    @torch.enable_grad()
    def loss_calculation(self, query_image=None, query_text=None, target_images_pool=None):
        self.reward_model.set_image_features(images=query_image)
        self.reward_model.set_text_features(captions=query_text)
        self.reward_model.set_mixture_features(mixture_factor_img=0.5, mixture_factor_text=0.5)
        teacher_text_factor = self.cfg.REWARD.TEACHER_TEXT_FACTOR
        self.reward_model.set_composed_features(mixture_factor_img=(1-teacher_text_factor), mixture_factor_text=teacher_text_factor)
        
        _, _, _, _, _, _, _, _, logits_per_query = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool)

        logits_per_query_teacher = self.reward_model.forward(self.reward_model.image_features, self.reward_model.text_features)

        loss = kd_distill_loss_v2(logits_per_query, logits_per_query_teacher)

        return loss


    @torch.enable_grad()
    def forward_and_adapt(self, query_image=None, query_text=None, target_images_pool=None):
        for _ in range(self.steps):
            if self.mixed_precision and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    loss = self.loss_calculation(query_image, query_text, target_images_pool)

                    with torch.no_grad():
                        query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            else:
                loss = self.loss_calculation(query_image, query_text, target_images_pool)
                
                with torch.no_grad():
                    query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features

    
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


