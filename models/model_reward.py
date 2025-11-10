import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from tqdm import tqdm
import logging
import torch
import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
# from models.third_party.open_clip import create_model_and_transforms, get_tokenizer

logger = logging.getLogger(__name__)



CONFIDECES = {
        "ViT-L-14": 10,
        "ViT-H-14": 10,
        "ViT-g-14": 10,
        "ViT-B-16": 1
    }

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    for p in model.parameters():
        p.data = p.data.half()
        if p.grad:
            p.grad.data = p.grad.data.half()


def get_reward_model(cfg, device):
    if cfg.REWARD.MULTIPLE_REWARD_MODELS:
        reward_model = CLIPRewardsMultiple(cfg, device, arch=["ViT-L-14", "ViT-H-14", "ViT-g-14"], 
                            pretrained_weights=["laion2b_s32b_b82k", "laion2b_s32b_b79k", "laion2b_s34b_b88k"], classification=True,
                            amplify_rewards=cfg.REWARD.REWARD_AMPLIFY, sample_k=cfg.REWARD.SAMPLE_K,
                            reward_process=cfg.REWARD.REWARD_PROCESS, process_batch=cfg.REWARD.PROCESS_BATCH,
                            weighted_scores=1)
    else:
        reward_model = CLIPRewards(cfg, device, arch=cfg.REWARD.REWARD_ARCH, classification=True,
                                amplify_rewards=cfg.REWARD.REWARD_AMPLIFY, sample_k=cfg.REWARD.SAMPLE_K,
                                reward_process=cfg.REWARD.REWARD_PROCESS, process_batch=cfg.REWARD.PROCESS_BATCH,
                                baseine_score=cfg.REWARD.BASEINE_SCORE, ema_decay=cfg.REWARD.MOMENTUM)

    return reward_model.to(device)


class BaseRewards(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_features = None
        self.image_features = None
        self.slerp_features = None
        self.nlerp_features = None
        self.reward_process = True
        self.amplify_rewards = False
        self.baseine_score = "mean"
        self.ema_decay = 0.9
        self.ema_baseline = 0

    @torch.no_grad()
    def extract_image_features(self, images):
        pass

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        pass

    @torch.no_grad()
    def set_text_features(self, captions=None, tokenized_cap=None, text_features=None):
        if text_features is None:
            self.text_features = self.extract_text_features(captions=captions, tokenized_cap=tokenized_cap)
        else:
            self.text_features = text_features

    @torch.no_grad()
    def set_image_features(self, images=None, image_features=None):
        if image_features is None:
            assert images is not None
            self.image_features = self.extract_image_features(images)
        else:
            self.image_features = image_features

    @torch.no_grad()
    def set_mixture_features(self, mixture_features=None, mixture_factor_img=0.5, mixture_factor_text=0.5):
        if mixture_features is None:
            assert self.text_features is not None and self.image_features is not None
            self.mixture_features = mixture_factor_img * self.image_features + mixture_factor_text * self.text_features 
        else:
            self.mixture_features = mixture_features


    @torch.no_grad()
    def set_composed_features(self, composed_features=None, mixture_factor_img=0.2, mixture_factor_text=0.8):
        if composed_features is None:
            assert self.text_features is not None and self.image_features is not None
            self.composed_features = mixture_factor_img * self.image_features + mixture_factor_text * self.text_features  
        else:
            self.composed_features = composed_features


    @torch.no_grad()
    def set_slerp_mixture_features(self, slerp_features=None, alpha=0.8, static_slerp=False):
        if slerp_features is None:
            assert self.text_features is not None and self.image_features is not None
            self.slerp_features = self.slerp_mixture(self.image_features, self.text_features, alpha=alpha, static_slerp=static_slerp)
        else:
            self.slerp_features = slerp_features

    @torch.no_grad()
    def set_nlerp_mixture_features(self, nlerp_features=None, alpha=0.8, epsilon=1.0, static_nlerp=False):
        if nlerp_features is None:
            assert self.text_features is not None and self.image_features is not None
            self.nlerp_features = self.nlerp_mixture(self.image_features, self.text_features, alpha=alpha, epsilon=epsilon, static_nlerp=static_nlerp)
        else:
            self.nlerp_features = nlerp_features


    @torch.no_grad()
    def slerp_mixture(self, v0, v1, alpha=0.8, static_slerp=False, delta=0.1, epsilon=1.0):
        if static_slerp:
            # Compute the cosine of the angle between the two vectors
            dot = (v0 * v1).sum(-1, keepdim=True)
            # If the dot product is close to 1, the vectors are nearly parallel
            # Use linear interpolation to avoid numerical precision issues
            close_condition = torch.abs(dot) > 0.9995
            linear_interp = v0 + alpha * (v1 - v0)
            linear_interp = linear_interp / linear_interp.norm(dim=-1, keepdim=True)
            # Compute the angle between the vectors and its sine
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            # Compute the scales for v0 and v1
            scale0 = torch.sin((1.0 - alpha) * theta) / sin_theta
            scale1 = torch.sin(alpha * theta) / sin_theta
            # Linearly interpolate between v0 and v1
            # This is equivalent to v0 * scale0 + v1 * scale1
            slerp_interp = scale0 * v0 + scale1 * v1
            # Normalize the output
            slerp_interp = slerp_interp / slerp_interp.norm(dim=-1, keepdim=True)
            # return torch.where(close_condition, linear_interp, slerp_interp)
        else:
            # Calculate the L2 distance between v0 and v1
            distance = torch.norm(v1 - v0, p=2, dim=-1, keepdim=True)

            # Calculate dynamic alpha (gamma_adapt)
            # alpha = alpha + delta * torch.tanh(distance / epsilon)
            alpha = torch.tanh(distance / epsilon)
            alpha = torch.clamp(alpha, min=0.0, max=1.0)

            # Compute the cosine of the angle between the two vectors
            dot = (v0 * v1).sum(-1, keepdim=True)
            close_condition = torch.abs(dot) > 0.9995
            
            dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            sin_theta_safe = sin_theta.clamp(min=1e-7)

            # Compute the scales for v0 and v1 using the dynamic alpha
            scale0 = torch.sin((1.0 - alpha) * theta) / sin_theta_safe
            scale1 = torch.sin(alpha * theta) / sin_theta_safe

            slerp_interp = scale0 * v0 + scale1 * v1
            slerp_interp = slerp_interp / slerp_interp.norm(dim=-1, keepdim=True)

        return slerp_interp

    @torch.no_grad()
    def nlerp_mixture(self, v0, v1, alpha=0.8, epsilon=1.0, static_nlerp=False, dis_type="l2"):
        if static_nlerp:
            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        else:
            if dis_type == "cosine":
                distance = 1 - F.cosine_similarity(v0, v1, dim=-1, eps=1e-8).unsqueeze(-1)
            elif dis_type == "l2":
                distance = torch.norm(v1 - v0, p=2, dim=-1, keepdim=True)
            elif dis_type == "frobenius":
                distance = torch.norm(v1 - v0, p='fro', dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid distance type: {dis_type}")

            alpha = torch.tanh(epsilon * distance)

            if self.cfg.CORRUPTION.DATASET == "cirr":
                alpha = torch.clamp(alpha, min=0.79, max=0.83)
            elif self.cfg.CORRUPTION.DATASET == "fashioniq":
                alpha = torch.clamp(alpha, min=0.65, max=0.75)
            else:
                alpha = torch.clamp(alpha, min=0.0, max=1.0)

            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        return nlerp_features


    @torch.no_grad()
    def confidence_gap(self, predictions):
        """
        Args:
            predictions: shape [bs, C]
        """
        value, index = torch.topk(predictions, 2, dim=-1)
        gap = value[:, 0] - value[:, 1]
        gap = gap - torch.mean(gap)

        return gap
    
    @torch.no_grad()
    def rewards_post_process(self, clip_score):  # Eq.(5)
        """
        clip_score: shape [bs, K] or [bs * K]
        """
        # if (clip_score.ndim > 1 and clip_score.shape[-1] > 1) or (clip_score.ndim == 1 and clip_score.shape[-1] > 1):
        if clip_score.shape[-1] > 1 and self.reward_process:


            mean = torch.mean(clip_score, dim=-1, keepdim=True)
            if self.amplify_rewards:
                std = torch.std(clip_score, dim=-1, keepdim=True) + 1e-5
            else:
                std = 1.0
            clip_score = (clip_score - mean) / std

        return clip_score.flatten()


class CLIPRewards(BaseRewards):
    def __init__(self, cfg, device, arch="ViT-B-16", pretrained="openai", clipscore_weight=2.5, classification=True,
                    amplify_rewards=False, sample_k=5, reward_process=True, process_batch=False, baseine_score="mean", ema_decay=0.995,
                    default_resolutions=224) -> None:
        """
        calculating CLIP Reward
        Args:
            clipscore_weight: weight for calculating CLIPScore
            reward_process: If ture, post-process the rewards, e.g., subtract the reward mean or standardization
            amplify_rewards: If true, after subtracting the reward mean, also divide rewards by standard variance of rewards, i.e, standardization.
            sample_k: K for sampling.
        """
        super().__init__()
        self.default_resolutions = default_resolutions

        self.clip_model, _, _ = create_model_and_transforms(cfg.REWARD.REWARD_ARCH,
                                                            pretrained=cfg.REWARD.REWARD_ARCH_WEIGHTS,
                                                            device=device,
                                                            precision=cfg.CLIP.PRECISION)
        convert_models_to_fp32(self.clip_model)
        # convert_models_to_fp16(self.clip_model)
        # self.clip_model.float()

        self.cfg = cfg
        self.resolutions = self.clip_model.visual.image_size
        self.clipscore_weight = cfg.REWARD.CLIP_SCORE_WEIGHT
        self.device = device
        self.classification = classification
        self.tokenize = get_tokenizer(cfg.REWARD.REWARD_ARCH)
        self.text_features = None
        self.image_features = None
        self.mixture_features = None
        self.mixture_factor_img = cfg.OPTIM.MIXTURE_FACTOR_IMG
        self.mixture_factor_text = cfg.OPTIM.MIXTURE_FACTOR_TEXT
        self.use_static_slerp = cfg.OPTIM.USE_STATIC_SLERP
        self.delta = cfg.OPTIM.DELTA
        self.epsilon = cfg.OPTIM.EPSILON
        self.fusion_type = cfg.OPTIM.FUSION_TYPE
        self.composed_features = None
        self.slerp_features = None
        self.target_image_features = None
        self.amplify_rewards = amplify_rewards
        self.sample_k = sample_k
        self.reward_process = reward_process
        self.process_batch = process_batch
        self.clip_model.eval()
        self.logit_scale = self.clip_model.logit_scale.exp()

        self.baseine_score = baseine_score
        self.ema_decay = ema_decay
        self.ema_baseline = torch.zeros(1).to(self.device)

        self.kurtosis_adjust = 0

        print("\n CLIPRewards model created: \n"
                "\t visual backbone: {}, resolutions: {}, amplify_rewards: {}, sample_k: {}, \n"
                "\t reward_process: {}, process_batch: {}\n".format(
                    arch, self.resolutions, amplify_rewards, sample_k, reward_process, process_batch))
    
    @torch.no_grad()
    def forward(self, query_image_features=None, query_text_features=None, target_image_features=None):
        if target_image_features is None:
            target_image_features = self.target_image_features

        mixture_features = self.mixture_features
        composed_features = self.composed_features 
        slerp_features = self.slerp_features
        nlerp_features = self.nlerp_features
        # if self.fusion_type == "slerp":
        #     logits_per_query = self.logit_scale * slerp_features @ target_image_features.t()
        # elif self.fusion_type == "nlerp":
        #     logits_per_query = self.logit_scale * nlerp_features @ target_image_features.t()
        # elif self.fusion_type == "lerp" or self.fusion_type == "avg":
        #     logits_per_query = self.logit_scale * composed_features @ target_image_features.t()

        logits_per_query = self.logit_scale * composed_features @ target_image_features.t() 
        
        return logits_per_query
    
    @torch.no_grad()
    def CLIPScore(self, query_index=None, target_index=None, pairwise=True):  # Eq.(4)
        """
        target_index: sampled target index
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # suitable for composed iamge retrieval, query_image_features + query_text_features = mixture_features => target_image_features

        self.sample_k = target_index.shape[0]
        
        if query_index is not None: 
            query_img_features = self.image_features[query_index]
            query_text_features = self.text_features[query_index]
            mixture_features = self.mixture_features[query_index]
            composed_features = self.composed_features[query_index]
            slerp_features = self.slerp_features[query_index]
            nlerp_features = self.nlerp_features[query_index]
        else:
            query_img_features = torch.repeat_interleave(self.image_features, self.sample_k, dim=0)
            query_text_features = torch.repeat_interleave(self.text_features, self.sample_k, dim=0)
            mixture_features = torch.repeat_interleave(self.mixture_features, self.sample_k, dim=0)
            composed_features = torch.repeat_interleave(self.composed_features, self.sample_k, dim=0)
            slerp_features = torch.repeat_interleave(self.slerp_features, self.sample_k, dim=0)
            nlerp_features = torch.repeat_interleave(self.nlerp_features, self.sample_k, dim=0)

        if target_index is not None:  
            target_image_features = self.target_image_features[target_index]
        else:
            target_image_features = torch.repeat_interleave(self.target_image_features, self.sample_k, dim=0)

        if pairwise:  
            similarity_img_only = self.clipscore_weight * query_img_features @ target_image_features.t()
            similarity_text_only = self.clipscore_weight * query_text_features @ target_image_features.t()
            similarity_mixture = self.clipscore_weight * mixture_features @ target_image_features.t()
            similarity_composed = self.clipscore_weight * composed_features @ target_image_features.t()
            similarity_slerp = self.clipscore_weight * slerp_features @ target_image_features.t()
            similarity_nlerp = self.clipscore_weight * nlerp_features @ target_image_features.t()
        else:
            similarity_img_only = self.clipscore_weight * torch.sum(query_img_features * target_image_features, dim=-1)
            similarity_text_only = self.clipscore_weight * torch.sum(query_text_features * target_image_features, dim=-1)
            similarity_mixture = self.clipscore_weight * torch.sum(mixture_features * target_image_features, dim=-1)
            similarity_composed = self.clipscore_weight * torch.sum(composed_features * target_image_features, dim=-1)
            similarity_slerp = self.clipscore_weight * torch.sum(slerp_features * target_image_features, dim=-1)
            similarity_nlerp = self.clipscore_weight * torch.sum(nlerp_features * target_image_features, dim=-1)

        scores_img_only = torch.maximum(similarity_img_only, torch.zeros_like(similarity_img_only)).squeeze()
        scores_text_only = torch.maximum(similarity_text_only, torch.zeros_like(similarity_text_only)).squeeze()
        scores_mixture = torch.maximum(similarity_mixture, torch.zeros_like(similarity_mixture)).squeeze()
        scores_composed = torch.maximum(similarity_composed, torch.zeros_like(similarity_composed)).squeeze()
        scores_slerp = torch.maximum(similarity_slerp, torch.zeros_like(similarity_slerp)).squeeze()
        scores_nlerp = torch.maximum(similarity_nlerp, torch.zeros_like(similarity_nlerp)).squeeze()


        return scores_img_only, scores_text_only, scores_mixture, scores_composed, scores_slerp, scores_nlerp


    @torch.no_grad()
    def CLIPScore_all(self, query_index=None, target_index=None, pairwise=True):  # Eq.(4)
        """
        target_index: sampled target index
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # suitable for composed iamge retrieval, query_image_features + query_text_features = mixture_features => target_image_features

        if query_index is not None: 
            query_img_features = self.image_features[query_index]
            query_text_features = self.text_features[query_index]
            mixture_features = self.mixture_features[query_index]
            composed_features = self.composed_features[query_index]
        else:
            query_img_features = torch.repeat_interleave(self.image_features, self.sample_k, dim=0)
            query_text_features = torch.repeat_interleave(self.text_features, self.sample_k, dim=0)
            mixture_features = torch.repeat_interleave(self.mixture_features, self.sample_k, dim=0)
            composed_features = torch.repeat_interleave(self.composed_features, self.sample_k, dim=0)

        if target_index is not None: 
            target_image_features = self.target_image_features[target_index]
        else:
            target_image_features = torch.repeat_interleave(self.target_image_features, self.sample_k, dim=0)

        if pairwise: 
            similarity_img_only = self.clipscore_weight * query_img_features @ target_image_features.t()
            similarity_text_only = self.clipscore_weight * query_text_features @ target_image_features.t()
            similarity_mixture = self.clipscore_weight * mixture_features @ target_image_features.t()
            similarity_composed = self.clipscore_weight * composed_features @ target_image_features.t()
        else:
            similarity_img_only = self.clipscore_weight * torch.sum(query_img_features * target_image_features, dim=-1)
            similarity_text_only = self.clipscore_weight * torch.sum(query_text_features * target_image_features, dim=-1)
            similarity_mixture = self.clipscore_weight * torch.sum(mixture_features * target_image_features, dim=-1)
            similarity_composed = self.clipscore_weight * torch.sum(composed_features * target_image_features, dim=-1)  # [32]

        scores_img_only = torch.maximum(similarity_img_only, torch.zeros_like(similarity_img_only)).squeeze()
        scores_text_only = torch.maximum(similarity_text_only, torch.zeros_like(similarity_text_only)).squeeze()
        scores_mixture = torch.maximum(similarity_mixture, torch.zeros_like(similarity_mixture)).squeeze()
        scores_composed = torch.maximum(similarity_composed, torch.zeros_like(similarity_composed)).squeeze()

        return scores_img_only, scores_text_only, scores_mixture, scores_composed


    
    @torch.no_grad()
    def rewards_post_process(self, clip_score, stu_probs=None, tea_probs=None): 
        """
        clip_score: shape [bs, K] or [bs * K]
        """
        if clip_score.shape[-1] > 1 and self.reward_process:
            original_clip_score = clip_score
            if self.baseine_score == "mean":
                mean = torch.mean(clip_score, dim=-1, keepdim=True)  
                clip_score = clip_score - mean  # (1, sample_k)
            
            elif self.baseine_score == "RLOO":
                sample_k = clip_score.shape[-1]  # K
                clip_score_exp = clip_score.expand(sample_k, sample_k)  # [K, K]        
                mask = torch.eye(sample_k, device=clip_score.device).bool()
                clip_score_exp = clip_score_exp.masked_fill(mask, 0)
                leave_one_out_mean = clip_score_exp.sum(dim=1) / (sample_k - 1)  
                self_score = clip_score.view(-1)  
                clip_score = self_score - leave_one_out_mean  


            if self.amplify_rewards:
                std = torch.std(original_clip_score, dim=-1, keepdim=True) + 1e-5  #      
            else:
                std = 1.0
            clip_score = clip_score / std
            

        return clip_score.flatten()


    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features with normalization"""
        if self.resolutions != self.default_resolutions:
            images = nn.functional.interpolate(images, size=self.resolutions, mode='bicubic', align_corners=True)
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        if captions is not None:
            tokenized_cap = self.tokenize(captions).to(self.device)
        
        text_features = self.clip_model.encode_text(tokenized_cap)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    # @torch.no_grad()
    # def set_many_text_features(self, texts, text_bs=128):
    #     """tokenize a list of 'texts' and call self.set_class_features()"""
    #     num_text = len(texts)
    #     text_feats = []
    #     i = 0
    #     while i < num_text:
    #         text = texts[i : min(num_text, i + text_bs)]
    #         input_ids = self.tokenize(text).to(self.device)[0]
    #         text_features = self.extract_text_features(tokenized_cap=input_ids)
    #         text_feats.append(text_features)
    #         i += text_bs

    #     self.text_features = torch.cat(text_feats, dim=0)

    @torch.no_grad()
    def set_image_features_with_dataloder(self, data_loader):
        image_feats = []
        for batch in tqdm(data_loader):  # 36
            image, _ = batch
            image = image.to(self.device)
            image_features = self.extract_image_features(image)
            image_feats.append(image_features)

        self.image_features = torch.cat(image_feats, dim=0)
        self.target_image_features = self.image_features  
   
    @torch.no_grad()
    def set_image_features_with_dataloder_coco(self, data_loader):
        image_feats = []
        for batch in tqdm(data_loader):  # 36
            image, _, _, _, _, _, _, _ = batch  
            image = image.to(self.device)
            image_features = self.extract_image_features(image)
            image_feats.append(image_features)

        self.image_features = torch.cat(image_feats, dim=0)
        self.target_image_features = self.image_features  
    

    @torch.no_grad()
    def set_image_features_with_dataloder_circo(self, data_loader):
        image_feats = []
        for batch in tqdm(data_loader):  # 36

            images = batch.get('image')
            names = batch.get('image_name')
            if images is None:
                images = batch.get('reference_image')
            if names is None:
                names = batch.get('reference_name')
            images = images.to(self.device)
            image_features = self.extract_image_features(images)
            image_feats.append(image_features)

        self.image_features = torch.cat(image_feats, dim=0)
        self.target_image_features = self.image_features  
   
    @torch.no_grad()
    def calulate_similarity(self):
        """
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * self.image_features @ self.text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class CLIPRewardsMultiple(BaseRewards):
    def __init__(self, cfg, device, arch=None, pretrained_weights="openai", precision="fp32",
                 clipscore_weight=2.5, classification=False,
                 amplify_rewards=False, sample_k=5, reward_process=True, process_batch=True, weighted_scores=True,
                 default_resolutions=224) -> None:
        """calculating CLIP Reward using multiple CLIP models
        Args:
            arch: a list of CLIP arches (e.g., ["ViT-B-16", "RN50x64"])
            pretrained_weights: name of pretrained weights for CLIP models (e.g., "openai", or a list matching arch)
            precision: precision for CLIP models (e.g., "fp32", "fp16", "amp")
            clipscore_weight: weight for calculating CLIPScore
            weighted_scores: if true, the final score is the weighted average of all scores
        """
        super().__init__()
        if arch is None:
            arch = ["ViT-B-16", "ViT-L-14", "ViT-H-14"] # Default architectures

        self.clip_models = nn.ModuleList()
        self.tokenizers = []
        self.resolutions = []
        model_weights_conf = []
        self.default_resolutions = default_resolutions
        self.device = device

        global CONFIDECES # Ensure we are using the global CONFIDECES
        self.logit_scales = [] # Initialize list for logit scales

        for i, model_arch_name in enumerate(arch):
            current_pretrained_weights = pretrained_weights[i] if isinstance(pretrained_weights, list) else pretrained_weights
            
            clip_model, _, _ = create_model_and_transforms(
                model_arch_name,
                pretrained=current_pretrained_weights,
                device=device, # device is passed here, model will be on specified device
                precision=precision 
            )
            
            # Ensure model is on the correct device, create_model_and_transforms might not move it if device=None
            clip_model = clip_model.to(device)

            if precision == "fp32":
                convert_models_to_fp32(clip_model)
            elif precision == "fp16":
                convert_models_to_fp16(clip_model)
            
            clip_model.eval() # Set model to evaluation mode

            self.clip_models.append(clip_model)
            self.tokenizers.append(get_tokenizer(model_arch_name))
            # open_clip models store image_size often as a tuple (H, W) or int. Take the first if tuple.
            vis_img_size = clip_model.visual.image_size
            self.resolutions.append(vis_img_size[0] if isinstance(vis_img_size, tuple) else vis_img_size)
            self.logit_scales.append(clip_model.logit_scale.exp()) # Store logit_scale
            
            if model_arch_name not in CONFIDECES:
                logger.warning(f"Model architecture {model_arch_name} not found in CONFIDECES. Using default weight 1.")
                model_weights_conf.append(1)
            else:
                model_weights_conf.append(CONFIDECES[model_arch_name])

        self.n_model = len(self.clip_models)
        if sum(model_weights_conf) == 0 and self.n_model > 0:
             self.weights = [1.0 / self.n_model] * self.n_model
        elif self.n_model == 0:
            self.weights = []
        else:
            self.weights = [round(x / sum(model_weights_conf), 2) for x in model_weights_conf]


        self.clipscore_weight = clipscore_weight
        self.classification = classification
        
        self.amplify_rewards = amplify_rewards # Inherited, but not directly used by this class's methods
        self.sample_k = sample_k
        self.reward_process = reward_process # Inherited, but not directly used by this class's methods
        self.process_batch = process_batch
        self.weighted_scores = weighted_scores

        self.text_features = None
        self.image_features = None
        self.mixture_features = None
        self.mixture_factor_img = cfg.OPTIM.MIXTURE_FACTOR_IMG
        self.mixture_factor_text = cfg.OPTIM.MIXTURE_FACTOR_TEXT
        self.use_static_slerp = cfg.OPTIM.USE_STATIC_SLERP
        self.delta = cfg.OPTIM.DELTA
        self.epsilon = cfg.OPTIM.EPSILON
        self.fusion_type = cfg.OPTIM.FUSION_TYPE
        self.composed_features = None
        self.slerp_features = None
        self.target_image_features = None

        logger.info("\nCLIPRewardsMultiple model created: \n"
                "\t visual backbones: {}, resolutions: {}, weighted_scores / weights: [ {} / {} ] \n"
                "\t amplify_rewards: {}, sample_k: {}, reward_process: {}, process_batch: {}\n".format(
                    arch, self.resolutions, self.weighted_scores, self.weights,
                    self.amplify_rewards, self.sample_k, self.reward_process, self.process_batch))

    @torch.no_grad()
    def CLIPScore(self, query_index=None, target_index=None, pairwise=True):
        """
        Calculates aggregated CLIPScore based on the fusion type using combined features.
        Args:
            query_index: Optional. Index/indices for selecting specific query features (e.g., from slerp_features).
            target_index: Optional. Index/indices for selecting specific target image features.
            pairwise: If True, computes all-pairs similarity. Else, computes element-wise similarity.
        Returns:
            Aggregated CLIPScore tensor.
        """
        if self.target_image_features is None:
            raise ValueError("Target image features must be set. Call set_image_features or set_image_features_with_dataloder first.")
        if not isinstance(self.target_image_features, list) or len(self.target_image_features) != self.n_model:
            raise ValueError("self.target_image_features should be a list of features, one per model.")


        query_features_list_source = None
        if self.fusion_type == "slerp":
            query_features_list_source = self.slerp_features
            if query_features_list_source is None: raise ValueError("Slerp features not set. Call set_slerp_mixture_features first.")
        elif self.fusion_type == "lerp":
            query_features_list_source = self.composed_features
            if query_features_list_source is None: raise ValueError("Composed features not set. Call set_composed_features first.")
        else:
            # If fusion_type is not for combined features, the original CLIPScore behavior might be intended.
            # The original user-provided CLIPScore used self.text_features and self.image_features.
            # For this modification, we are focusing on composed query functionality.
            # If a different behavior is needed for other fusion_types, this needs explicit handling.
            # Fallback to text_features vs target_image_features (image-text scoring)
            logger.warning(f"CLIPScore called with fusion_type '{self.fusion_type}'. Using text_features as query vs target_image_features.")
            query_features_list_source = self.text_features
            if query_features_list_source is None: raise ValueError("Text features not set for default CLIPScore behavior.")

        if not isinstance(query_features_list_source, list) or len(query_features_list_source) != self.n_model:
            raise ValueError(f"Selected query features source is not a valid list of {self.n_model} tensors.")
            
        all_model_scores = []
        for i in range(self.n_model):
            current_query_features = query_features_list_source[i]
            current_target_img_features = self.target_image_features[i]

            if current_query_features is None:
                logger.warning(f"Model {i}: Query features for fusion type '{self.fusion_type}' are None. Skipping CLIPScore calculation.")
                all_model_scores.append(None) # Placeholder
                continue
            if current_target_img_features is None:
                logger.warning(f"Model {i}: Target image features are None. Skipping CLIPScore calculation.")
                all_model_scores.append(None)
                continue

            # Handle indexing and repeating for sampling.
            # The original CLIPRewardsMultiple.CLIPScore used self.sample_k for repeating if index was None.
            # We adopt a similar strategy: if an index is None, repeat source by self.sample_k.
            # If an index is provided, use it directly (don't repeat by self.sample_k unless that index itself implies multiple items).

            q_features = current_query_features[query_index] if query_index is not None else \
                            torch.repeat_interleave(current_query_features, self.sample_k, dim=0)
            
            # For target features, if pairwise, we might use all targets if target_index is None.
            # If not pairwise, target should align with query samples.
            if pairwise:
                t_features = current_target_img_features[target_index] if target_index is not None else \
                                current_target_img_features # All targets if target_index is None
            else: # Element-wise
                # If query was repeated by sample_k, target should also be repeated if target_index is None.
                if query_index is None: # Query was repeated
                    t_features = current_target_img_features[target_index] if target_index is not None else \
                                 torch.repeat_interleave(current_target_img_features, self.sample_k, dim=0)
                else: # Query was specifically indexed
                    t_features = current_target_img_features[target_index] if target_index is not None else \
                                 current_target_img_features # Assume it aligns or target_index is specific
            
            if pairwise:
                # q_features: [N_q_samples, D], t_features: [N_t_samples, D] -> similarity: [N_q_samples, N_t_samples]
                similarity = self.clipscore_weight * q_features @ t_features.t()
            else:
                # q_features: [N_samples, D], t_features: [N_samples, D] -> similarity: [N_samples]
                similarity = self.clipscore_weight * torch.sum(q_features * t_features, dim=-1)

            scores_i = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()
            all_model_scores.append(scores_i)

        # Filter out None scores and aggregate
        processed_scores = [s for s in all_model_scores if s is not None]
        if not processed_scores:
            logger.warning("No model produced valid scores for CLIPScore.")
            if all_model_scores and all_model_scores[0] is not None:
                 return torch.empty(0, dtype=all_model_scores[0].dtype, device=self.device)
            return torch.empty(0, device=self.device)

        # Ensure all tensors in processed_scores are on the same device for stacking
        processed_scores = [s.to(self.device) for s in processed_scores]

        if len(processed_scores) == 1:
            final_scores = processed_scores[0]
        else:
            try:
                stacked_scores = torch.stack(processed_scores, dim=0)
            except RuntimeError as e:
                logger.error(f"Error stacking scores during CLIPScore: {e}. Score shapes: {[s.shape for s in processed_scores]}")
                raise e

            if self.weighted_scores and len(self.weights) == self.n_model:
                active_model_indices = [idx for idx, score in enumerate(all_model_scores) if score is not None]
                active_weights = [self.weights[idx] for idx in active_model_indices]
                active_weights_sum = sum(active_weights)
                
                if active_weights_sum > 1e-6:
                    normalized_active_weights = torch.tensor([w / active_weights_sum for w in active_weights], device=stacked_scores.device, dtype=stacked_scores.dtype)
                    while normalized_active_weights.ndim < stacked_scores.ndim:
                        normalized_active_weights = normalized_active_weights.unsqueeze(-1)
                    final_scores = torch.sum(normalized_active_weights * stacked_scores, dim=0)
                else:
                    logger.warning("Sum of active model weights for CLIPScore is near zero. Falling back to mean aggregation.")
                    final_scores = torch.mean(stacked_scores, dim=0)
            else:
                final_scores = torch.mean(stacked_scores, dim=0)

        return final_scores

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features with normalization for each model"""
        image_features_list = []
        images_on_device = images.to(self.device)
        for i in range(self.n_model):
            # Handle resolution differences
            if self.resolutions[i] != self.default_resolutions: # Assuming default_resolutions is a single int
                # Check if images need resizing based on their current size vs target model resolution
                # This simplistic check might need refinement if images can have varying initial sizes.
                # For now, we assume images are at default_resolutions or need resizing to self.resolutions[i].
                # The `interpolate` function expects H, W. self.resolutions[i] should be the target size.
                current_model_resolution = (self.resolutions[i], self.resolutions[i]) if isinstance(self.resolutions[i], int) else self.resolutions[i]
                if images_on_device.shape[-2:] != current_model_resolution:
                    tmp_images = nn.functional.interpolate(images_on_device, size=current_model_resolution, mode='bicubic', align_corners=True)
                else:
                    tmp_images = images_on_device
            else:
                tmp_images = images_on_device
            
            image_feat = self.clip_models[i].encode_image(tmp_images) # .float() is handled by convert_models_to_fpX or precision
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            image_features_list.append(image_feat)
        return image_features_list # List of tensors [batch_size, dim]

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        """extract text features with normalization for each model"""
        text_features_list = []
        if captions is None and tokenized_cap is None:
            raise ValueError("Either captions or tokenized_cap must be provided.")

        for i in range(self.n_model):
            if captions is not None:
                # Tokenize captions for the current model
                caption_tokens = self.tokenizers[i](captions).to(self.device)
                text_feat = self.clip_models[i].encode_text(caption_tokens)
            elif tokenized_cap is not None:
                # If tokenized_cap is provided, it's assumed to be a single tensor compatible with all models,
                # or it should be a list of tokenized inputs (one per model).
                # For simplicity, if it's a single tensor:
                current_tokenized_cap = tokenized_cap.to(self.device) if torch.is_tensor(tokenized_cap) else tokenized_cap[i].to(self.device)
                text_feat = self.clip_models[i].encode_text(current_tokenized_cap)
            
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            text_features_list.append(text_feat)
        return text_features_list # List of tensors [batch_size, dim]

    @torch.no_grad()
    def set_text_features(self, captions=None, tokenized_cap=None, text_features=None):
        """Sets self.text_features. Input can be raw captions, pre-tokenized captions, or pre-computed features."""
        if text_features is not None:
            # text_features should be a list of tensors, one for each model
            if not isinstance(text_features, list) or len(text_features) != self.n_model:
                raise ValueError(f"Provided text_features must be a list of {self.n_model} tensors.")
            self.text_features = [feat.to(self.device) for feat in text_features]
        elif captions is not None or tokenized_cap is not None:
            self.text_features = self.extract_text_features(captions=captions, tokenized_cap=tokenized_cap)
        else:
            raise ValueError("Must provide captions, tokenized_cap, or text_features.")

    @torch.no_grad()
    def set_image_features(self, images=None, image_features=None):
        """Sets self.image_features. Input can be raw images or pre-computed features."""
        if image_features is not None:
            # image_features should be a list of tensors, one for each model
            if not isinstance(image_features, list) or len(image_features) != self.n_model:
                raise ValueError(f"Provided image_features must be a list of {self.n_model} tensors.")
            self.image_features = [feat.to(self.device) for feat in image_features]
        elif images is not None:
            self.image_features = self.extract_image_features(images)
        else:
            raise ValueError("Must provide images or image_features.")


    @torch.no_grad()
    def set_many_text_features(self, texts, text_bs=128):
        """Tokenize a list of 'texts' and set self.text_features for all models."""
        num_text = len(texts)
        # Initialize list of lists: outer list for models, inner for text batches
        batched_text_feats_per_model = [[] for _ in range(self.n_model)]
        
        i = 0
        while i < num_text:
            text_batch = texts[i : min(num_text, i + text_bs)]
            for model_idx in range(self.n_model):
                tokenized_batch = self.tokenizers[model_idx](text_batch).to(self.device)
                current_model_text_features = self.clip_models[model_idx].encode_text(tokenized_batch)
                current_model_text_features = current_model_text_features / current_model_text_features.norm(dim=-1, keepdim=True)
                batched_text_feats_per_model[model_idx].append(current_model_text_features)
            i += text_bs

        # Concatenate features for each model
        final_text_features_per_model = []
        for model_idx in range(self.n_model):
            if batched_text_feats_per_model[model_idx]:
                final_text_features_per_model.append(torch.cat(batched_text_feats_per_model[model_idx], dim=0))
            else:
                # Handle case with no texts or empty batches if necessary
                # Create an empty tensor with correct feature dimension if known, or None
                # For now, assume clip_models[0] exists if n_model > 0
                feat_dim = self.clip_models[model_idx].text_projection.shape[-1] if self.n_model > 0 and hasattr(self.clip_models[model_idx], 'text_projection') and self.clip_models[model_idx].text_projection is not None else 512 # Fallback
                empty_tensor = torch.empty((0, feat_dim), device=self.device, dtype=self.clip_models[model_idx].dtype if self.n_model > 0 else torch.float32)
                final_text_features_per_model.append(empty_tensor)


        self.text_features = final_text_features_per_model

    @torch.no_grad()
    def set_image_features_with_dataloder(self, data_loader):
        """Extract image features for all models using a DataLoader and set self.image_features."""
        # Initialize list of lists: outer list for models, inner for image batches
        batched_image_feats_per_model = [[] for _ in range(self.n_model)]

        for batch_data in tqdm(data_loader, desc="Extracting image features for multiple models"):
            # Assuming data_loader yields batches of images, or dicts containing images
            if isinstance(batch_data, dict):
                images = batch_data.get('image', batch_data.get('reference_image'))
                if images is None:
                    logger.warning("Key 'image' or 'reference_image' not found in batch from DataLoader. Skipping batch.")
                    continue
            elif torch.is_tensor(batch_data):
                images = batch_data
            elif isinstance(batch_data, (list, tuple)) and torch.is_tensor(batch_data[0]): # common case: (image_batch, label_batch)
                 images = batch_data[0]
            else:
                logger.warning(f"Unsupported batch data type from DataLoader: {type(batch_data)}. Skipping batch.")
                continue

            images = images.to(self.device)
            # extract_image_features returns a list of feature tensors (one per model for this batch)
            current_batch_image_features_list = self.extract_image_features(images)
            
            for model_idx in range(self.n_model):
                batched_image_feats_per_model[model_idx].append(current_batch_image_features_list[model_idx])

        # Concatenate features for each model
        final_image_features_per_model = []
        for model_idx in range(self.n_model):
            if batched_image_feats_per_model[model_idx]:
                final_image_features_per_model.append(torch.cat(batched_image_feats_per_model[model_idx], dim=0))
            else:
                feat_dim = self.clip_models[model_idx].visual.output_dim if self.n_model > 0 else 512 # Fallback
                empty_tensor = torch.empty((0, feat_dim), device=self.device, dtype=self.clip_models[model_idx].dtype if self.n_model > 0 else torch.float32)
                final_image_features_per_model.append(empty_tensor)

        self.image_features = final_image_features_per_model
        self.target_image_features = final_image_features_per_model # Mirror image_features for target features
        
    @torch.no_grad()
    def set_mixture_features(self, mixture_features_list=None, mixture_factor_img=None, mixture_factor_text=None):
        if mixture_features_list is not None:
            if not isinstance(mixture_features_list, list) or len(mixture_features_list) != self.n_model:
                raise ValueError(f"Provided mixture_features_list must be a list of {self.n_model} tensors.")
            self.mixture_features = [feat.to(self.device) for feat in mixture_features_list]
            return

        if self.text_features is None or self.image_features is None:
            raise ValueError("Text and image features must be set before computing mixture features.")
        if len(self.text_features) != self.n_model or len(self.image_features) != self.n_model:
            raise ValueError("Mismatch in the number of model features for text or image.")

        factor_img = mixture_factor_img if mixture_factor_img is not None else self.mixture_factor_img
        factor_text = mixture_factor_text if mixture_factor_text is not None else self.mixture_factor_text
        
        self.mixture_features = []
        for i in range(self.n_model):
            if self.image_features[i] is None or self.text_features[i] is None:
                self.mixture_features.append(None)
                logger.warning(f"Model {i}: Image or text features are None. Cannot compute mixture features.")
                continue
            
            img_feat_device = self.image_features[i].to(self.device)
            txt_feat_device = self.text_features[i].to(self.device)
            mix_feat = factor_img * img_feat_device + factor_text * txt_feat_device
            self.mixture_features.append(mix_feat / mix_feat.norm(dim=-1, keepdim=True))

    @torch.no_grad()
    def set_composed_features(self, composed_features_list=None, mixture_factor_img=None, mixture_factor_text=None):
        if composed_features_list is not None:
            if not isinstance(composed_features_list, list) or len(composed_features_list) != self.n_model:
                raise ValueError(f"Provided composed_features_list must be a list of {self.n_model} tensors.")
            self.composed_features = [feat.to(self.device) for feat in composed_features_list]
            return

        if self.text_features is None or self.image_features is None:
            raise ValueError("Text and image features must be set before computing composed features.")
        if len(self.text_features) != self.n_model or len(self.image_features) != self.n_model:
             raise ValueError("Mismatch in the number of model features for text or image.")

        factor_img = mixture_factor_img
        factor_text = mixture_factor_text

        self.composed_features = []
        for i in range(self.n_model):
            if self.image_features[i] is None or self.text_features[i] is None:
                self.composed_features.append(None)
                logger.warning(f"Model {i}: Image or text features are None. Cannot compute composed features.")
                continue

            img_feat_device = self.image_features[i].to(self.device)
            txt_feat_device = self.text_features[i].to(self.device)
            comp_feat = factor_img * img_feat_device + factor_text * txt_feat_device
            self.composed_features.append(comp_feat / comp_feat.norm(dim=-1, keepdim=True))
            
    @torch.no_grad()
    def set_slerp_mixture_features(self, slerp_features_list=None, alpha=None, static_slerp=None):
        if slerp_features_list is not None:
            if not isinstance(slerp_features_list, list) or len(slerp_features_list) != self.n_model:
                raise ValueError(f"Provided slerp_features_list must be a list of {self.n_model} tensors.")
            self.slerp_features = [feat.to(self.device) for feat in slerp_features_list]
            return

        if self.text_features is None or self.image_features is None:
            raise ValueError("Text and image features must be set before computing SLERP features.")
        if len(self.text_features) != self.n_model or len(self.image_features) != self.n_model:
             raise ValueError("Mismatch in the number of model features for text or image.")

        alpha_to_use = alpha if alpha is not None else self.alpha_slerp
        static_slerp_to_use = static_slerp if static_slerp is not None else self.use_static_slerp

        self.slerp_features = []
        for i in range(self.n_model):
            if self.image_features[i] is None or self.text_features[i] is None:
                self.slerp_features.append(None)
                logger.warning(f"Model {i}: Image or text features are None. Cannot compute SLERP features.")
                continue
            
            # slerp_mixture is inherited from BaseRewards and expects features on the same device.
            img_feat_device = self.image_features[i].to(self.device)
            txt_feat_device = self.text_features[i].to(self.device)

            slerp_feat = self.slerp_mixture(
                img_feat_device, 
                txt_feat_device, 
                alpha=alpha_to_use, 
                static_slerp=static_slerp_to_use,
                delta=self.delta,
                epsilon=self.epsilon
            )
            self.slerp_features.append(slerp_feat) # slerp_mixture from BaseRewards already normalizes

    @torch.no_grad()
    def forward(self, query_index=None, target_index=None, pairwise=True):
        """
        Calculates aggregated logits based on the fusion type.
        Handles multiple models, similar to how CLIPScore will, but for logits.
        Args:
            query_index: Optional. Index/indices for selecting specific query features.
            target_index: Optional. Index/indices for selecting specific target features.
            pairwise: If True, computes all-pairs similarity. Else, computes element-wise similarity.
        Returns:
            Aggregated logits tensor.
        """
        if self.target_image_features is None:
            raise ValueError("Target image features must be set. Call set_image_features or set_image_features_with_dataloder first.")
        if not isinstance(self.target_image_features, list) or len(self.target_image_features) != self.n_model:
            raise ValueError("self.target_image_features should be a list of features, one per model.")

        query_features_list_source = None
        if self.fusion_type == "slerp":
            query_features_list_source = self.slerp_features
            if query_features_list_source is None: raise ValueError("Slerp features not set. Call set_slerp_mixture_features first.")
        elif self.fusion_type == "lerp":
            query_features_list_source = self.composed_features
            if query_features_list_source is None: raise ValueError("Composed features not set. Call set_composed_features first.")
        else:
            # Fallback or error for undefined fusion_type for composed query. 
            # If standard image-text or text-image is intended, this logic might differ.
            # For now, assuming forward() is primarily for composed queries like in CLIPRewards.
            raise ValueError(f"Unsupported fusion_type for forward: {self.fusion_type}. Expected 'slerp', 'mixture', or 'composed'.")
        
        if not isinstance(query_features_list_source, list) or len(query_features_list_source) != self.n_model:
            raise ValueError(f"Selected query features source is not a valid list of {self.n_model} tensors.")

        all_model_logits = []
        for i in range(self.n_model):
            current_query_features = query_features_list_source[i]
            current_target_img_features = self.target_image_features[i]
            current_logit_scale = self.logit_scales[i]

            if current_query_features is None:
                logger.warning(f"Model {i}: Query features for fusion type '{self.fusion_type}' are None. Skipping logit calculation.")
                all_model_logits.append(None) # Placeholder to maintain list length for aggregation
                continue
            if current_target_img_features is None:
                logger.warning(f"Model {i}: Target image features are None. Skipping logit calculation.")
                all_model_logits.append(None)
                continue

            # Select query samples
            # If query_index is None, and we expect self.sample_k repetition, this logic must match CLIPScore.
            # CLIPRewards.forward doesn't use self.sample_k for repeating, it expects features to be already prepared.
            # For consistency with CLIPRewards.forward, let's assume features are directly used or indexed.
            q_features = current_query_features[query_index] if query_index is not None else current_query_features
            
            # Select target samples
            t_features = current_target_img_features[target_index] if target_index is not None else current_target_img_features

            if pairwise:
                # q_features: [N_q, D], t_features: [N_t, D] -> logits: [N_q, N_t]
                logits_i = current_logit_scale * q_features @ t_features.t()
            else:
                # q_features: [N, D], t_features: [N, D] -> logits: [N]
                logits_i = current_logit_scale * torch.sum(q_features * t_features, dim=-1)
            all_model_logits.append(logits_i)
        
        # Filter out None logits if any model failed
        processed_logits = [l for l in all_model_logits if l is not None]
        if not processed_logits:
            logger.warning("No model produced valid logits.")
            # Determine what to return: empty tensor, raise error, etc.
            # For now, let's try to return an empty tensor of appropriate type if possible or raise
            if all_model_logits and all_model_logits[0] is not None: # Get dtype from a valid logit if exists
                 return torch.empty(0, dtype=all_model_logits[0].dtype, device=self.device)
            return torch.empty(0, device=self.device)

        # Aggregate logits
        # Ensure all tensors in processed_logits are on the same device for stacking
        processed_logits = [l.to(self.device) for l in processed_logits]
        
        # If only one model produced logits, no need to stack/aggregate further than just returning it.
        if len(processed_logits) == 1:
            final_logits = processed_logits[0]
        else:
            try:
                stacked_logits = torch.stack(processed_logits, dim=0)
            except RuntimeError as e:
                logger.error(f"Error stacking logits during forward pass: {e}. Logit shapes: {[l.shape for l in processed_logits]}")
                # Attempt to handle non-uniform batch sizes if that's the cause and aggregation is still possible (e.g. sum/mean of scalar rewards)
                # For now, re-raise if stacking fails as shapes should be consistent for typical batch operations.
                raise e

            if self.weighted_scores and len(self.weights) == self.n_model: # Ensure weights match original model count
                # We need to select weights corresponding to models that produced logits.
                # This requires knowing which original models correspond to processed_logits.
                # Simpler: if some models failed, weighted average might be skewed if weights aren't adjusted.
                # For now, assume all models contribute or use mean if weighted is tricky with failures.
                # A more robust way: filter weights according to non-None logits.
                active_model_indices = [idx for idx, logit in enumerate(all_model_logits) if logit is not None]
                active_weights = [self.weights[idx] for idx in active_model_indices]
                active_weights_sum = sum(active_weights)
                
                if active_weights_sum > 1e-6 : # Avoid division by zero
                    normalized_active_weights = torch.tensor([w / active_weights_sum for w in active_weights], device=stacked_logits.device, dtype=stacked_logits.dtype)
                    while normalized_active_weights.ndim < stacked_logits.ndim:
                        normalized_active_weights = normalized_active_weights.unsqueeze(-1)
                    final_logits = torch.sum(normalized_active_weights * stacked_logits, dim=0)
                else: # Fallback to mean if all active weights are zero (should not happen with positive weights)
                    logger.warning("Sum of active model weights is near zero. Falling back to mean aggregation.")
                    final_logits = torch.mean(stacked_logits, dim=0)
            else:
                final_logits = torch.mean(stacked_logits, dim=0)

        return final_logits

