"""
Builds upon: https://github.com/oripress/CCC/blob/main/models/rpl.py
Corresponding paper: https://arxiv.org/pdf/2104.12928.pdf
"""

import torch
import torch.nn as nn
from scipy import stats
import numpy as np

from methods.base import TTAMethod_with_reward
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import GeneralizedCrossEntropy
import logging
logger = logging.getLogger(__name__)

@ADAPTATION_REGISTRY.register()
class Reward(TTAMethod_with_reward):
    """
    Reward model for test-time adaptation.
    """
    def __init__(self, cfg, model, reward_model, combiner, num_classes):
        super().__init__(cfg, model, reward_model, combiner, num_classes)
        
        self.cfg = cfg


    def sample_k_index(self, logits_per_query, sample_k, H_score=None):
        if self.cfg.REWARD.SAMPLING_TYPE == "topk": 
            value, index = torch.topk(logits_per_query, sample_k, dim=-1)
            target_index = index.flatten()
        elif self.cfg.REWARD.SAMPLING_TYPE == "temperature": 
            temperature = self.cfg.REWARD.TEMPERATURE 
            scaled_logits = logits_per_query / temperature
            probs = torch.softmax(scaled_logits, dim=-1) 

            probs = probs * H_score

            target_index = torch.multinomial(probs, sample_k) 
            target_index = target_index.flatten()
        elif self.cfg.REWARD.SAMPLING_TYPE == "random":
            target_index = torch.randint(0, logits_per_query.size(1), (sample_k,), device=logits_per_query.device)
            target_index = target_index.flatten()

        else:
            raise ValueError(f"Unknown sampling type: {self.cfg.REWARD.SAMPLING_TYPE}")

        return target_index

    def cal_H_score(self, logits_per_query, logits_query_text_counterfactual):
        # normalize
        logits_per_query = logits_per_query/logits_per_query.norm(dim=-1, keepdim=True)
        logits_query_text_counterfactual = logits_query_text_counterfactual/logits_query_text_counterfactual.norm(dim=-1, keepdim=True)
        H_score = logits_per_query/(logits_query_text_counterfactual+1e-6)

        H_score_mask = (H_score > 1)
        H_score = torch.where(H_score_mask, torch.ones_like(H_score), H_score)
        H_score = torch.where(~H_score_mask, 1e-6*torch.ones_like(H_score), H_score)

        return H_score

    @torch.enable_grad()
    def loss_calculation(self, query_image=None, query_text=None, target_images_pool=None):
        self.reward_model.set_image_features(images=query_image)
        self.reward_model.set_text_features(captions=query_text)
        self.reward_model.set_mixture_features(mixture_factor_img=0.5, mixture_factor_text=0.5)
        teacher_text_factor = self.cfg.REWARD.TEACHER_TEXT_FACTOR
        self.reward_model.set_composed_features(mixture_factor_img=(1-teacher_text_factor), mixture_factor_text=teacher_text_factor)
        
        if not self.cfg.REWARD.MULTIPLE_REWARD_MODELS:
            self.reward_model.set_slerp_mixture_features (alpha=self.reward_model.mixture_factor_text, static_slerp=self.reward_model.use_static_slerp)
            self.reward_model.set_nlerp_mixture_features(alpha=self.reward_model.mixture_factor_text, epsilon=self.reward_model.epsilon, static_nlerp=self.reward_model.use_static_slerp)

        # query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_images_features, logits_per_query, logits_query_image, logits_query_text = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool, need_target=True, need_all_logits=True)  # (1, gallery_size)
        query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_images_features, logits_per_query, logits_query_image, logits_query_text, logits_query_text_counterfactual = self.model.forward_with_perturb(images=query_image, text=query_text, target_images=target_images_pool, need_target=True, need_all_logits=True)  # (1, gallery_size)

        # hard socre
        H_score = self.cal_H_score(logits_per_query, logits_query_text_counterfactual)
        # logger.info(f"H_score: {H_score}")

        sample_k = self.cfg.REWARD.SAMPLE_K
        target_index = self.sample_k_index(logits_per_query, sample_k, H_score)

        if self.cfg.REWARD.MULTIPLE_REWARD_MODELS:
            clip_score = self.reward_model.CLIPScore(query_index=None, target_index=target_index, pairwise=False)
        else:
            if self.reward_model.fusion_type == "slerp":
                img_socre, text_socre, _, _, clip_score,_ = self.reward_model.CLIPScore(query_index=None, target_index=target_index, pairwise=False)  # slerp_log
            elif self.reward_model.fusion_type == "nlerp":
                img_socre, text_socre, _, _, _, clip_score = self.reward_model.CLIPScore(query_index=None, target_index=target_index, pairwise=False)  # nlerp_log
            elif self.reward_model.fusion_type == "lerp" or self.reward_model.fusion_type == "avg":
                img_socre, text_socre, _, clip_score, _, _= self.reward_model.CLIPScore(query_index=None, target_index=target_index, pairwise=False)  # 2/8_log

        com_rewards = self.reward_model.rewards_post_process(
            clip_score if self.reward_model.process_batch else clip_score.reshape(query_image.size(0), -1),
        )  # (sample_k)
        com_rewards_ref = com_rewards.clone()

        if self.cfg.REWARD.USE_SELF_REWARD:
            img_socre_self, text_socre_self, _, clip_score_self, _, _ = self.CLIPScore_self(query_index=None, target_index=target_index, pairwise=False, img_features=query_image_features, text_features=query_text_features, mixture_features=mixture_features, lerp_features=lerp_features, nlerp_features=nlerp_features, slerp_features=slerp_features, target_images_features=target_images_features, sample_k=sample_k)
            com_rewards_self = self.reward_model.rewards_post_process(clip_score_self if self.reward_model.process_batch else clip_score_self.reshape(query_image.size(0), -1))
            com_rewards = com_rewards + self.cfg.REWARD.SELF_REWARD_WEIGHT * com_rewards_self

 
        rewards = com_rewards.squeeze()

        rep_output = torch.repeat_interleave(logits_per_query, sample_k, dim=0)  # (sample_k, gallery_size)

        use_penalty = False
        if self.cfg.REWARD.USE_SELF_REWARD and use_penalty:    
            if self.cfg.CORRUPTION.DATASET == "fashioniq" and self.cfg.CORRUPTION.DATASET_FIQTYPE == "dress":
                rewards = rewards
            else:
                positive_mask = (com_rewards_ref.squeeze() > 0) & (com_rewards_self.squeeze() > 0)
                rewards = torch.where(positive_mask, rewards, -torch.ones_like(rewards))
        
        all_loss = nn.functional.cross_entropy(rep_output, target_index, reduction='none')
        ce_loss = torch.mean(rewards * all_loss) 
        loss = ce_loss
    
        return logits_per_query, loss, rewards


    @torch.enable_grad()
    def forward_and_adapt(self, query_image=None, query_text=None, target_images_pool=None):
        prev_rewards = None
        for i in range(self.steps):
            if self.mixed_precision and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    logits_per_query, loss, rewards = self.loss_calculation(query_image, query_text, target_images_pool)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()                

            else:
                logits_per_query, loss, rewards = self.loss_calculation(query_image, query_text, target_images_pool)
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        with torch.no_grad():
            query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, _ = self.model.forward(images=query_image, text=query_text, target_images=target_images_pool)


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
                    if np in ['weight', 'bias'] and p.requires_grad and ("attention" not in f"{nm}.{np}" and "output_query" not in f"{nm}.{np}"):
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
        for nm, m in self.model.named_modules():   
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
            # elif 'combiner' in str(type(m).__name__).lower():
            #     m.requires_grad_(True)



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

    @torch.no_grad()
    def CLIPScore_self(self, query_index=None, target_index=None, pairwise=True, img_features=None, text_features=None, mixture_features=None, lerp_features=None, nlerp_features=None, slerp_features=None, target_images_features=None, clipscore_weight=2.5, sample_k=32):  # Eq.(4)
        """
        target_index: sampled target index
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # suitable for composed iamge retrieval, query_image_features + query_text_features = mixture_features => target_image_features
        
        if query_index is not None:
            query_img_features = img_features[query_index]
            query_text_features = text_features[query_index]
            mixture_features = mixture_features[query_index]
            lerp_features = lerp_features[query_index]
            nlerp_features = nlerp_features[query_index]
            slerp_features = slerp_features[query_index]
        else:
            query_img_features = torch.repeat_interleave(img_features, sample_k, dim=0)
            query_text_features = torch.repeat_interleave(text_features, sample_k, dim=0)
            mixture_features = torch.repeat_interleave(mixture_features, sample_k, dim=0)
            lerp_features = torch.repeat_interleave(lerp_features, sample_k, dim=0)
            nlerp_features = torch.repeat_interleave(nlerp_features, sample_k, dim=0)
            slerp_features = torch.repeat_interleave(slerp_features, sample_k, dim=0)

        if target_index is not None:  
            target_image_features = target_images_features[target_index]
        else:
            target_image_features = torch.repeat_interleave(target_images_features, sample_k, dim=0)

        if pairwise:  
            similarity_img_only = clipscore_weight * query_img_features @ target_image_features.t()
            similarity_text_only = clipscore_weight * query_text_features @ target_image_features.t()
            similarity_mixture = clipscore_weight * mixture_features @ target_image_features.t()
            similarity_lerp = clipscore_weight * lerp_features @ target_image_features.t()
            similarity_nlerp = clipscore_weight * nlerp_features @ target_image_features.t()
            similarity_slerp = clipscore_weight * slerp_features @ target_image_features.t()
        else:
            similarity_img_only = clipscore_weight * torch.sum(query_img_features * target_image_features, dim=-1)
            similarity_text_only = clipscore_weight * torch.sum(query_text_features * target_image_features, dim=-1)
            similarity_mixture = clipscore_weight * torch.sum(mixture_features * target_image_features, dim=-1)
            similarity_lerp = clipscore_weight * torch.sum(lerp_features * target_image_features, dim=-1)
            similarity_nlerp = clipscore_weight * torch.sum(nlerp_features * target_image_features, dim=-1)
            similarity_slerp = clipscore_weight * torch.sum(slerp_features * target_image_features, dim=-1)

        scores_img_only = torch.maximum(similarity_img_only, torch.zeros_like(similarity_img_only)).squeeze()
        scores_text_only = torch.maximum(similarity_text_only, torch.zeros_like(similarity_text_only)).squeeze()
        scores_mixture = torch.maximum(similarity_mixture, torch.zeros_like(similarity_mixture)).squeeze()
        scores_lerp = torch.maximum(similarity_lerp, torch.zeros_like(similarity_lerp)).squeeze()
        scores_nlerp = torch.maximum(similarity_nlerp, torch.zeros_like(similarity_nlerp)).squeeze()
        scores_slerp = torch.maximum(similarity_slerp, torch.zeros_like(similarity_slerp)).squeeze()


        return scores_img_only, scores_text_only, scores_mixture, scores_lerp, scores_nlerp, scores_slerp


