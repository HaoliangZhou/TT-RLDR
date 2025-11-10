import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import logging
import math
import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import spacy
import en_core_web_sm
import random

from models.third_party.open_clip import create_model_and_transforms, get_tokenizer

from typing import Union
from copy import deepcopy
from models.custom_clip import ClipTestTimePromptTuning
from models.third_party.lavis.models import load_model_and_preprocess as lavis_load_model_and_preprocess
from models.third_party.lavis.models.base_model import all_gather_with_grad, concat_all_gather
from packaging import version
from datasets.cls_names import get_class_names
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
import copy


logger = logging.getLogger(__name__)

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

def kl_divergence(p, q, epsilon=1e-8):
    p_safe = p.clamp(min=epsilon)
    q_safe = q.clamp(min=epsilon)
    return (p_safe * (p_safe.log() - q_safe.log())).sum(dim=-1)


def get_torchvision_model(model_name: str, weight_version: str = "IMAGENET1K_V1"):
    """
    Restore a pre-trained model from torchvision
    Further details can be found here: https://pytorch.org/vision/0.14/models.html
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
        weight_version: Name of the pre-trained weights to restore
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    assert version.parse(torchvision.__version__) >= version.parse("0.13"), "Torchvision version has to be >= 0.13"

    # check if the specified model name is available in torchvision
    available_models = torchvision.models.list_models(module=torchvision.models)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision. Choose from: {available_models}")

    # get the weight object of the specified model and the available weight initialization names
    model_weights = torchvision.models.get_model_weights(model_name)
    available_weights = [init_name for init_name in dir(model_weights) if "IMAGENET1K" in init_name]

    # check if the specified type of weights is available
    if weight_version not in available_weights:
        raise ValueError(f"Weight type '{weight_version}' is not supported for torchvision model '{model_name}'."
                         f" Choose from: {available_weights}")

    # restore the specified weights
    model_weights = getattr(model_weights, weight_version)

    # setup the specified model and initialize it with the specified pre-trained weights
    model = torchvision.models.get_model(model_name, weights=model_weights)

    # get the transformation and add the input normalization to the model
    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    logger.info(f"Successfully restored '{weight_version}' pre-trained weights"
                f" for model '{model_name}' from torchvision!")

    # create the corresponding input transformation
    preprocess = transforms.Compose([transforms.Resize(transform.resize_size, interpolation=transform.interpolation),
                                     transforms.CenterCrop(transform.crop_size),
                                     transforms.ToTensor()])
    return model, preprocess


def get_timm_model(model_name: str):
    """
    Restore a pre-trained model from timm: https://github.com/huggingface/pytorch-image-models/tree/main/timm
    Quickstart: https://huggingface.co/docs/timm/quickstart
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    # check if the defined model name is supported as pre-trained model
    available_models = timm.list_models(pretrained=True)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in timm. Choose from: {available_models}")

    # setup pre-trained model
    model = timm.create_model(model_name, pretrained=True)
    logger.info(f"Successfully restored the weights of '{model_name}' from timm.")

    # restore the input pre-processing
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config)

    # if there is an input normalization, add it to the model and remove it from the input pre-processing
    for transf in preprocess.transforms[::-1]:
        if isinstance(transf, transforms.Normalize):
            # add input normalization to the model
            model = normalize_model(model, mean=transf.mean, std=transf.std)
            preprocess.transforms.remove(transf)
            break

    return model, preprocess





class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias) 
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)  
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)  
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, img_feat, text_feat):
        B, N, C = img_feat.shape
        _, M, _ = text_feat.shape
        
        q = self.q_proj(img_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(text_feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(text_feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, img_feat, text_feat):
        x = img_feat
        attn_output, attn = self.attn(self.norm1(x), text_feat)
        x = x + self.drop_path(attn_output)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x[:,0,:], attn



class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    def forward(self, image_features: torch.tensor, text_features: torch.tensor, target_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)
        target_features = F.normalize(target_features, dim=-1)

        logits = self.logit_scale * predicted_features @ target_features.T
        return logits

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
        # return F.normalize(output, dim=-1)
        return output

class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)



class ZeroShotCLIP(nn.Module):
    def __init__(self, cfg, model, device, normalize):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.device = device
        self.visual = model.visual
        self.normalize = normalize
        self.prompt_mode = cfg.CLIP.PROMPT_MODE
        self.freeze_text_encoder = cfg.CLIP.FREEZE_TEXT_ENCODER
        self.class_names = get_class_names(cfg.CORRUPTION.DATASET)

        self.vocab_size = model.vocab_size
        self.end_id = self.vocab_size - 1
        self.tokenize = get_tokenizer(cfg.MODEL.ARCH)
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.text_projection = model.text_projection
        self.ln_final = model.ln_final
        self.logit_scale = self.model.logit_scale.data

        # prevents test-time adaptation methods from unfreezing parameters in the text encoder
        # if self.freeze_text_encoder:
        #     self.model.transformer = None

    @property
    def dtype(self):
        return next(self.model.visual.parameters()).dtype

    def encode_image(self, image):
        image_feat, _ = self.visual(image.type(self.dtype))
        return image_feat

    def encode_image_tokens(self, image):
        return self.visual.get_tokens(image.type(self.dtype))

    def encode_text(self, text=None, tokenized_prompts=None):
        if tokenized_prompts is None:
            assert text is not None
            tokenized_prompts = self.tokenize(text).to(self.device)
        x, x_local = self.model.encode_text(tokenized_prompts)
        return x

    def encode_text_img(self, text, img_tokens):
        b_size = img_tokens.size(0)
        tokenized_text = self.tokenize(text).to(self.device)[0]
        x = self.model.encode_text(tokenized_text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id
        collect_ind = collect_ind.nonzero()[:, 1]
        img_tokens = img_tokens.view(b_size, 1, -1)
        x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-1]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind + 1] @ self.text_projection
        return x

    def encode_text_img_replace_test(self, text, img_tokens):
        b_size = img_tokens.size(0)
        tokenized_text = self.tokenize(text).to(self.device)[0]
        x = self.model.encode_text(tokenized_text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id
        collect_ind = collect_ind.nonzero()[:, 1]

        collect_inds_replace = text == self.replace_id
        collect_inds_replace = collect_inds_replace.nonzero()[:, 1]

        img_tokens = img_tokens.view(b_size, 1, -1)
        x = torch.cat([x[:, :collect_inds_replace[0]], img_tokens, x[:, collect_inds_replace[0]:-1]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x

    def encode_text_img_replace(self, text, img_tokens):
        b_size = img_tokens.size(0)
        tokenized_text = self.tokenize(text).to(self.device)[0]
        x = self.model.encode_text(tokenized_text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id
        collect_ind = collect_ind.nonzero()[:, 1]
        collect_inds_replace = text == self.replace_id

        # replace the word embedding by image token embedding
        # there are different number of word tokens in each caption which should be replaced
        for batch_idx in range(len(collect_inds_replace)):
            collect_ind_replace_list = collect_inds_replace[batch_idx].nonzero()[:, 0]
            # print(collect_ind_replace_list)
            for collect_ind_replace in collect_ind_replace_list:
                x[batch_idx][collect_ind_replace] = img_tokens[batch_idx]
                # print(img_tokens[batch_idx])
        # collect_inds_replace = collect_inds_replace.nonzero()[:, 1]
        # img_tokens = img_tokens.view(b_size, 1, -1)
        # x = torch.cat([x[:, :collect_inds_replace[0]], img_tokens, x[:, collect_inds_replace[0]:-1]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection  # without add image tokens
        return x

    def encode_text_img_vis(self, text, img_tokens, split_ind=4):
        tokenized_text = self.tokenize(text).to(self.device)[0]
        x = self.model.encode_text(tokenized_text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id
        collect_ind = collect_ind.nonzero()[:, 1]
        new_x = []
        for i, sample in enumerate(x):
            ind_insert = text[i] == split_ind
            sample = sample.view(1, x.size(1), -1)
            if isinstance(img_tokens, tuple):
                indexes = ind_insert.nonzero()
                for i, index in enumerate(indexes):
                    img = img_tokens[i].view(1, 1, -1)
                    sample = torch.cat([sample[:, :index], img, sample[:, index + 1:]], dim=1)
            else:
                img_tokens = img_tokens.view(1, 1, -1)
                ind_insert = ind_insert.nonzero()[0]
                sample = torch.cat([sample[:, :ind_insert], img_tokens, sample[:, ind_insert + 1:]], dim=1)
            new_x.append(sample)
        x = torch.cat(new_x, dim=0)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x

    def encode_text_img_retrieval(self, text, img_tokens, split_ind=4, repeat=True):
        # text.shape = [1, n_ctx]
        # img_tokens.shape = [batch_size, d_model]
        if isinstance(img_tokens, tuple):
            b_size = img_tokens[0].shape[0]
        else:
            b_size = img_tokens.shape[0]
        if repeat:
            text = text.repeat(b_size, 1)  # (bs, 77)
        tokenized_text = self.tokenize(text).to(self.device)[0]
        x = self.model.encode_text(tokenized_text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id
        collect_ind = collect_ind.nonzero()[:, 1]  # (bs,)
        ind_insert = text[0] == split_ind
        if isinstance(img_tokens, tuple):
            indexes = ind_insert.nonzero()
            for i, index in enumerate(indexes):
                img = img_tokens[i].view(b_size, 1, -1)
                x = torch.cat([x[:, :index], img, x[:, index + 1:]], dim=1)
        else:
            img_tokens = img_tokens.view(b_size, 1, -1)
            ind_insert = ind_insert.nonzero()[0]
            x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert + 1:]], dim=1)
        # x = torch.cat([x, torch.zeros_like(x).cuda()[:, :1, :]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x
    

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


    def nlerp_mixture(self, v0, v1, alpha=0.8, epsilon=1.0, static_nlerp=False):
        if static_nlerp:
            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        else:
            distance = torch.norm(v1 - v0, p=2, dim=-1, keepdim=True)
            alpha = torch.tanh(distance / epsilon)

            if self.cfg.CORRUPTION.DATASET == "cirr":
                alpha = torch.clamp(alpha, min=0.79, max=0.83)
            elif self.cfg.CORRUPTION.DATASET == "fashioniq":
                alpha = torch.clamp(alpha, min=0.65, max=0.75)
            else:
                alpha = torch.clamp(alpha, min=0.0, max=1.0)

            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        return nlerp_features


    def forward(self, image, text, extra=False):
        if image is None:
            if extra:
                return self.encode_text_extra(text)
            else:
                return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        if extra:
            text_features = self.encode_text_extra(text)
        else:
            text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.logit_scale.exp()

    
class CLIPRet_TTA(nn.Module):
    """
    Using CLIP to do image retrieval with Test Time Adaptation,
    this module will tune the whole or part parameters of CLIP image encoder or text encoder.
    Args:
        only_norm: if True, only update the normalization layer
    """
    def __init__(self, cfg, model, combiner, device):
        super().__init__()
        self.cfg = cfg
        self.clip_model = model
        self.combiner = combiner
        self.cross_attention_fusion = CrossAttentionFusion(dim=768)
        self.tokenize = get_tokenizer(cfg.MODEL.ARCH)
        self.device = device
        self.text_features = None
        self.image_features = None
        self.mixture_features = None
        self.mixture_factor_img = cfg.OPTIM.MIXTURE_FACTOR_IMG
        self.mixture_factor_text = cfg.OPTIM.MIXTURE_FACTOR_TEXT
        self.use_static_slerp = cfg.OPTIM.USE_STATIC_SLERP
        self.delta = cfg.OPTIM.DELTA
        self.epsilon = cfg.OPTIM.EPSILON
        self.fusion_type = cfg.OPTIM.FUSION_TYPE
        self.slerp_features = None
        self.composed_features = None
        self.target_image_features = None
        self.target_covariance = None
        self.proj_matrix = None
        self.momentum_update = cfg.REWARD.MOMENTUM_UPDATE
        self.update_freq = cfg.REWARD.UPDATE_FREQ
        self.update_w = cfg.REWARD.UPDATE_W
        self.momentum = cfg.REWARD.MOMENTUM
        self.update_counter = 0
        self.perturbation_std = 0.01

        # save model state of visual encoder
        with torch.no_grad():
            self.clip_state_dict = copy.deepcopy(self.clip_model.state_dict())
            self.initial_state_dict = copy.deepcopy(self.clip_model.state_dict())
            if self.momentum_update:
                self.momentum_state_dict = copy.deepcopy(self.clip_model.state_dict())


        print("\n CLIPRet_TTA model created: \n"
                "\t backbone: {}, momentum_update / momentum / update_freq / update_w: [{} / {} / {} / {}] \n".format(
                    cfg.MODEL.ARCH, self.momentum_update, self.momentum, self.update_freq, self.update_w))


    def forward(self, images=None, text=None, tokenized_prompts=None, target_images=None, need_target=False, need_all_logits=False):     
        query_image_features, query_image_local_features = self.get_image_features(images, Global_only=False) if images is not None else self.image_features
        query_text_features, query_text_local_features = self.get_text_features(text, tokenized_prompts, Global_only=False) if text is not None or tokenized_prompts is not None else self.text_features
        target_image_features = self.get_target_image_features_pool(target_images) if target_images is not None else self.target_image_features

        mixture_features = 0.5 * query_image_features + 0.5 * query_text_features
        mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)

        lerp_features = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features
        lerp_features = lerp_features / lerp_features.norm(dim=-1, keepdim=True)

        nlerp_features = self.nlerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_nlerp=self.use_static_slerp, epsilon=self.epsilon)

        slerp_features = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=self.use_static_slerp, delta=self.delta, epsilon=self.epsilon)

        slerp_features_static = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=True)

        # composed_features = lerp_features
        composed_features,_ = self.cross_attention_fusion(torch.cat((query_image_features.unsqueeze(1), query_image_local_features), dim=1), query_text_local_features)
        composed_features = composed_features / composed_features.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()


        if self.fusion_type == "slerp":
            logits_per_query = logit_scale * slerp_features @ target_image_features.t()
        elif self.fusion_type == "nlerp":
            logits_per_query = logit_scale * nlerp_features @ target_image_features.t()
        elif self.fusion_type == "avg":
            logits_per_query = logit_scale * mixture_features @ target_image_features.t()
        elif self.fusion_type == "lerp":
            logits_per_query = logit_scale * lerp_features @ target_image_features.t()
        elif self.fusion_type == "composed":
            logits_per_query = logit_scale * composed_features @ target_image_features.t()

        
        if need_target and not need_all_logits:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query 
        elif need_target and need_all_logits:
            logits_query_image = logit_scale * query_image_features @ target_image_features.t()
            logits_query_text = logit_scale * query_text_features @ target_image_features.t()
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query, logits_query_image, logits_query_text
        else:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, logits_per_query 


    def forward_with_perturb(self, images=None, text=None, tokenized_prompts=None, target_images=None, need_target=False, need_all_logits=False):     
        query_image_features, query_image_local_features = self.get_image_features(images, Global_only=False) if images is not None else self.image_features
        query_text_features, query_text_local_features = self.get_text_features(text, tokenized_prompts, Global_only=False) if text is not None or tokenized_prompts is not None else self.text_features
        target_image_features = self.get_target_image_features_pool(target_images) if target_images is not None else self.target_image_features

        mixture_features = 0.5 * query_image_features + 0.5 * query_text_features
        mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)

        lerp_features = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features
        lerp_features = lerp_features / lerp_features.norm(dim=-1, keepdim=True)

        nlerp_features = self.nlerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_nlerp=self.use_static_slerp, epsilon=self.epsilon)

        slerp_features = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=self.use_static_slerp, delta=self.delta, epsilon=self.epsilon)

        slerp_features_static = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=True)

        composed_features = lerp_features

        logit_scale = self.clip_model.logit_scale.exp()

        new_text = self.get_parsing_for_text(text)
        query_text_features_counterfactual= self.get_text_features(new_text, tokenized_prompts=None, Global_only=True)

        lerp_features_counterfactual = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features_counterfactual
        lerp_features_counterfactual = lerp_features_counterfactual / lerp_features_counterfactual.norm(dim=-1, keepdim=True)

        if self.fusion_type == "slerp":
            logits_per_query = logit_scale * slerp_features @ target_image_features.t()
        elif self.fusion_type == "nlerp":
            logits_per_query = logit_scale * nlerp_features @ target_image_features.t()
        elif self.fusion_type == "avg":
            logits_per_query = logit_scale * mixture_features @ target_image_features.t()
        elif self.fusion_type == "lerp":
            logits_per_query = logit_scale * lerp_features @ target_image_features.t()
        elif self.fusion_type == "composed":
            logits_per_query = logit_scale * composed_features @ target_image_features.t()

        
        if need_target and not need_all_logits:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query 
        elif need_target and need_all_logits:
            logits_query_image = logit_scale * query_image_features @ target_image_features.t()
            logits_query_text = logit_scale * query_text_features @ target_image_features.t()
            logits_query_text_counterfactual = logit_scale * lerp_features_counterfactual @ target_image_features.t()
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query, logits_query_image, logits_query_text, logits_query_text_counterfactual
        else:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, logits_per_query 

    def get_parsing_for_text(self, text):
        if isinstance(text, tuple) or isinstance(text, list):
            text = text[0]
        
        if not isinstance(text, str):
            raise ValueError(f"Expected string or tuple of strings, but got {type(text)}")
        
        nlp = en_core_web_sm.load()
        doc = nlp(text)
        nouns = [w for w in doc if w.pos_ == "NOUN"]
        verbs = [w for w in doc if w.pos_ == "VERB"]
        adjs = [w for w in doc if w.pos_ == "ADJ"]
        advs = [w for w in doc if w.pos_ == "ADV"]

        parsing = nouns + verbs + adjs + advs

        if not parsing:
            return text

        mask_more_words = True
        if mask_more_words: 
            new_text = text
            for word in parsing:
                new_text = new_text.replace(word.text, '[MASK]')
        else:
            random_word = random.choice(parsing)
            new_text = text.replace(random_word.text, '[MASK]')
        return new_text
    
        
    def get_text_features(self, text=None, tokenized_prompts=None, Global_only=True):
        if tokenized_prompts is None:
            assert text is not None
            tokenized_prompts = self.tokenize(text).to(self.device)
        if Global_only:
            text_features, _ = self.clip_model.encode_text(tokenized_prompts)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            return text_features
        else:
            text_features, text_local_features = self.clip_model.encode_text(tokenized_prompts)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            text_local_features = text_local_features/text_local_features.norm(dim=-1, keepdim=True)
            return text_features, text_local_features
   

    def get_image_features(self, images, Global_only=True):
        if Global_only:
            image_features, _ = self.clip_model.encode_image(images)
            # image_features = F.normalize(image_features, dim=-1)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            return image_features
        else:
            image_features, local_features = self.clip_model.encode_image(images)  # image_features: [bs,512], local_features: [bs,196,512]
            # image_features = F.normalize(image_features, dim=-1)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            local_features = local_features/local_features.norm(dim=-1, keepdim=True)
            return image_features, local_features

    
    def get_target_image_features_pool(self, target_images):
        target_image_features = target_images
        return target_image_features

    def get_mixture_features(self, text, image):
        tokenized_text = self.tokenize(text).to(self.device)[0]
        query_text_features = self.clip_model.encode_text(tokenized_text)
        query_image_features = self.clip_model.encode_image(image, Global_only=True)
        mixture_features = self.mixture_factor_img * query_image_features + self.mixture_factor_text * query_text_features
        # mixture_features = F.normalize(mixture_features, dim=-1)
        mixture_features = mixture_features/mixture_features.norm(dim=-1, keepdim=True)
        return mixture_features

    def get_slerp_features(self, query_image_features, query_text_features, alpha=0.8, static_slerp=False):
        slerp_features = self.slerp_mixture(query_image_features, query_text_features, alpha=alpha, static_slerp=static_slerp)
        return slerp_features

    def get_composed_features(self, query_image_features, query_text_features):
        composed_features = self.combiner.combine_features(query_image_features, query_text_features)
        # composed_features = F.normalize(composed_features, dim=-1)
        composed_features = composed_features/composed_features.norm(dim=-1, keepdim=True)
        return composed_features


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
    

    def nlerp_mixture(self, v0, v1, alpha=0.8, epsilon=1.0, static_nlerp=False, dis_type="frobenius"):
        if static_nlerp:
            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        else:
            if dis_type == "cosine":
                distance = F.cosine_similarity(v0, v1, dim=-1, eps=1e-8).unsqueeze(-1)
            elif dis_type == "l2":
                distance = torch.norm(v1 - v0, p=2, dim=-1, keepdim=True)
            elif dis_type == "frobenius":
                distance = torch.norm(v1 - v0, p='fro', dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid distance type: {dis_type}")

            alpha = torch.tanh(epsilon * distance)
            alpha = torch.clamp(alpha, min=0.79, max=0.83)

            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        return nlerp_features

    
    def set_image_features(self, images=None, image_features=None, Global_only=True):
        if images is not None:
            self.image_features = self.get_image_features(images, Global_only=Global_only)
        else:
            self.image_features = image_features

    def set_text_features(self, text=None, tokenized_prompts=None, text_features=None):
        if text is not None or tokenized_prompts is not None:
            self.text_features = self.get_text_features(text, tokenized_prompts, Global_only=True)
        else:
            assert text_features is not None
            self.text_features = text_features

    def set_mixture_features(self, mixture_features=None):
        if mixture_features is None:
            assert self.text_features is not None and self.image_features is not None
            self.mixture_features = self.mixture_factor_img * self.image_features + self.mixture_factor_text * self.text_features
            # self.mixture_features = F.normalize(self.mixture_features, dim=-1)
        else:
            self.mixture_features = mixture_features


    def set_slerp_features(self, slerp_features=None):
        if slerp_features is None:
            assert self.text_features is not None and self.image_features is not None
            self.slerp_features = self.slerp_mixture(self.image_features, self.text_features, alpha=0.9, static_slerp=False)
        else:
            self.slerp_features = slerp_features
    
    def set_nlerp_features(self, nlerp_features=None):
        if nlerp_features is None:
            assert self.text_features is not None and self.image_features is not None
            self.nlerp_features = self.nlerp_mixture(self.image_features, self.text_features, alpha=0.9, static_nlerp=False)
        else:
            self.nlerp_features = nlerp_features

    def set_composed_features(self, composed_features=None):
        if composed_features is None:
            assert self.text_features is not None and self.image_features is not None
            self.composed_features = self.combiner.combine_features(self.image_features, self.text_features)
            # self.composed_features = F.normalize(self.composed_features, dim=-1)
        else:
            self.composed_features = composed_features

    @torch.no_grad()
    def reset_all(self):
        # reset the state dict of clip model, sometimes you may change the weights
        self.clip_model.load_state_dict(self.clip_state_dict)
        self.initial_state_dict = copy.deepcopy(self.clip_model.state_dict())
        if self.momentum_update:
            self.momentum_state_dict = copy.deepcopy(self.clip_model.state_dict())

    @torch.no_grad()
    def reset_initial(self):
        self.clip_model.load_state_dict(self.initial_state_dict)

    @torch.no_grad()
    def momentum_update_model(self):
        update_w = self.update_w
        if self.momentum_update:
            self.update_counter += 1
            # reload momentum state_dict
            state_dict = self.clip_model.state_dict()
            for k, v in state_dict.items():
                self.momentum_state_dict[k]= self.momentum * self.momentum_state_dict[k] + (1.0 - self.momentum) * v

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                for k, v in state_dict.items():
                    # self.initial_state_dict[k] = self.momentum_state_dict[k]
                    self.initial_state_dict[k] = (1 - update_w) * self.clip_state_dict[k] + update_w * self.momentum_state_dict[k]
            # update will be done by function self.reset_initial()



class ZeroShotBLIP2(nn.Module):
    def __init__(self, cfg, model, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        self.visual_encoder = model.visual_encoder
        self.ln_vision = model.ln_vision
        self.Qformer = model.Qformer
        self.query_tokens = model.query_tokens
        self.vision_proj = model.vision_proj
        self.text_proj = model.text_proj
        self.tokenize = model.tokenizer

        self.mixture_factor_img = cfg.OPTIM.MIXTURE_FACTOR_IMG
        self.mixture_factor_text = cfg.OPTIM.MIXTURE_FACTOR_TEXT
        self.use_static_slerp = cfg.OPTIM.USE_STATIC_SLERP
        self.fusion_type = cfg.OPTIM.FUSION_TYPE
        self.delta = cfg.OPTIM.DELTA
        self.epsilon = cfg.OPTIM.EPSILON
        
        self.logit_scale = torch.tensor(100.0)

        self.image_features = None
        self.text_features = None
        self.mixture_features = None
        self.slerp_features = None
        self.composed_features = None
        self.target_image_features = None

    def encode_image(self, image):
        image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state[:, 0, :]
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        image_features = concat_all_gather(image_features)

        return image_features

    def encode_text(self, text):
        text = self.tokenize(text, return_tensors="pt", padding=True).to(self.device)
        
        text_output = self.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        text_features = F.normalize(self.text_proj(text_features), dim=-1)

        text_features = concat_all_gather(text_features)

        return text_features

    def extract_features(self, samples, mode="multimodal"):
        image = samples.get("image")
        text = samples.get("text_input")
        
        image_features, text_features, multimodal_features = None, None, None
        
        if mode == "image":
            assert image is not None, "Image is required for image mode"
            image_features = self.encode_image(image)
            
        elif mode == "text":
            assert text is not None, "Text is required for text mode"
            text_features = self.encode_text(text)
            
        elif mode == "multimodal":
            assert image is not None and text is not None, "Both image and text required for multimodal mode"
            
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            
            text_tokens = self.tokenize(text, return_tensors="pt", padding=True).to(self.device)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            
            output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            multimodal_features = output.last_hidden_state[:, :query_tokens.size(1), :]
            multimodal_features = F.normalize(self.vision_proj(multimodal_features), dim=-1)

        return {
            "image_features": image_features,
            "text_features": text_features,
            "multimodal_features": multimodal_features
        }


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
    

    def nlerp_mixture(self, v0, v1, alpha=0.8, epsilon=1.0, static_nlerp=False, dis_type="frobenius"):
        if static_nlerp:
            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        else:
            if dis_type == "cosine":
                distance = F.cosine_similarity(v0, v1, dim=-1, eps=1e-8).unsqueeze(-1)
            elif dis_type == "l2":
                distance = torch.norm(v1 - v0, p=2, dim=-1, keepdim=True)
            elif dis_type == "frobenius":
                distance = torch.norm(v1 - v0, p='fro', dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid distance type: {dis_type}")

            alpha = torch.tanh(epsilon * distance)
            alpha = torch.clamp(alpha, min=0.79, max=0.83)

            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        return nlerp_features


    def forward(self, images=None, text=None, target_images=None, need_target=False, need_all_logits=False):     
        query_image_features = self.encode_image(images) if images is not None else self.image_features
        query_text_features = self.encode_text(text) if text is not None else self.text_features
        target_image_features = self.get_target_image_features_pool(target_images) if target_images is not None else self.target_image_features

        mixture_features = 0.5 * query_image_features + 0.5 * query_text_features
        mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)

        lerp_features = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features
        lerp_features = lerp_features / lerp_features.norm(dim=-1, keepdim=True)

        nlerp_features = self.nlerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_nlerp=self.use_static_slerp, epsilon=self.epsilon)

        slerp_features = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=self.use_static_slerp, delta=self.delta, epsilon=self.epsilon)

        slerp_features_static = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=True)

        composed_features = lerp_features

        logit_scale = self.logit_scale

        if self.fusion_type == "slerp":
            logits_per_query = logit_scale * slerp_features @ target_image_features.t()
        elif self.fusion_type == "avg":
            logits_per_query = logit_scale * mixture_features @ target_image_features.t()
        elif self.fusion_type == "lerp":
            logits_per_query = logit_scale * lerp_features @ target_image_features.t()

        
        if need_target and not need_all_logits:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query 
        elif need_target and need_all_logits:
            logits_query_image = logit_scale * query_image_features @ target_image_features.t()
            logits_query_text = logit_scale * query_text_features @ target_image_features.t()
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query, logits_query_image, logits_query_text
        else:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, logits_per_query 


    def forward_with_perturb(self, images=None, text=None, target_images=None, need_target=False, need_all_logits=False):     
        query_image_features = self.encode_image(images) if images is not None else self.image_features
        query_text_features = self.encode_text(text) if text is not None else self.text_features
        target_image_features = self.get_target_image_features_pool(target_images) if target_images is not None else self.target_image_features

        mixture_features = 0.5 * query_image_features + 0.5 * query_text_features
        mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)

        lerp_features = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features
        lerp_features = lerp_features / lerp_features.norm(dim=-1, keepdim=True)

        nlerp_features = self.nlerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_nlerp=self.use_static_slerp, epsilon=self.epsilon)

        slerp_features = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=self.use_static_slerp, delta=self.delta, epsilon=self.epsilon)

        slerp_features_static = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=True)

        composed_features = lerp_features

        new_text = self.get_parsing_for_text(text)
        query_text_features_counterfactual= self.encode_text(new_text)

        lerp_features_counterfactual = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features_counterfactual
        lerp_features_counterfactual = lerp_features_counterfactual / lerp_features_counterfactual.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale

        if self.fusion_type == "slerp":
            logits_per_query = logit_scale * slerp_features @ target_image_features.t()
        elif self.fusion_type == "avg":
            logits_per_query = logit_scale * mixture_features @ target_image_features.t()
        elif self.fusion_type == "lerp":
            logits_per_query = logit_scale * lerp_features @ target_image_features.t()

        
        if need_target and not need_all_logits:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query 
        elif need_target and need_all_logits:
            logits_query_image = logit_scale * query_image_features @ target_image_features.t()
            logits_query_text = logit_scale * query_text_features @ target_image_features.t()
            logits_query_text_counterfactual = logit_scale * lerp_features_counterfactual @ target_image_features.t()
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query, logits_query_image, logits_query_text, logits_query_text_counterfactual
        else:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, logits_per_query 

    def get_parsing_for_text(self, text):
        if isinstance(text, tuple) or isinstance(text, list):
            text = text[0]
        
        if not isinstance(text, str):
            raise ValueError(f"Expected string or tuple of strings, but got {type(text)}")
        
        nlp = en_core_web_sm.load()
        doc = nlp(text)
        nouns = [w for w in doc if w.pos_ == "NOUN"]
        verbs = [w for w in doc if w.pos_ == "VERB"]
        adjs = [w for w in doc if w.pos_ == "ADJ"]
        advs = [w for w in doc if w.pos_ == "ADV"]

        parsing = nouns + verbs + adjs + advs

        if not parsing:
            return text

        mask_more_words = True
        if mask_more_words: 
            new_text = text
            for word in parsing:
                new_text = new_text.replace(word.text, '[MASK]')
        else:
            random_word = random.choice(parsing)
            new_text = text.replace(random_word.text, '[MASK]')
        return new_text

    def get_text_features(self, text=None, tokenized_prompts=None):
        if tokenized_prompts is None:
            text_features = self.encode_text(text)
        else:
            text_features = self.encode_text(tokenized_prompts)
        return text_features


    def get_image_features(self, images, Global_only=True):
        image_features = self.encode_image(images)
        return image_features  # encode_image already normalizes


    def get_target_image_features_pool(self, target_images):
        # Assuming target_images are already processed features
        target_image_features = target_images
        self.target_image_features = target_image_features # Cache it
        return target_image_features

    def get_mixture_features(self, text, image):
        # This method assumes single text and image input for calculation
        # It doesn't use cached self.text_features / self.image_features by default
        query_text_features = self.get_text_features(text=text)
        query_image_features = self.get_image_features(images=image)
        mixture_features = self.mixture_factor_img * query_image_features + self.mixture_factor_text * query_text_features
        mixture_features = mixture_features/mixture_features.norm(dim=-1, keepdim=True)
        return mixture_features

    def get_slerp_features(self, query_image_features, query_text_features, alpha=None, static_slerp=False, delta=None, epsilon=None):
        # Uses cached features if available, otherwise requires features as input
        img_feat = query_image_features if query_image_features is not None else self.image_features
        txt_feat = query_text_features if query_text_features is not None else self.text_features
        
        if img_feat is None or txt_feat is None:
             raise ValueError("Image and Text features must be provided or set before calling get_slerp_features.")

        slerp_features = self.slerp_mixture(img_feat, txt_feat, alpha=alpha, static_slerp=static_slerp, delta=delta, epsilon=epsilon)
        # slerp_mixture already normalizes
        return slerp_features

    def get_composed_features(self, query_image_features, query_text_features):
         raise NotImplementedError("ZeroShotBLIP2 does not have a 'combiner' module integrated.")


    def set_image_features(self, images=None, image_features=None, Global_only=True):
        if images is not None:
            self.image_features = self.encode_image(images)
        elif image_features is not None:
            # Assume pre-computed features are already normalized if passed directly
            self.image_features = image_features
        else:
            raise ValueError("Either images or image_features must be provided.")

    def set_text_features(self, text=None, text_features=None):
        if text is not None:
            self.text_features = self.encode_text(text)
        elif text_features is not None:
             # Assume pre-computed features are already normalized if passed directly
            self.text_features = text_features
        else:
             raise ValueError("Either text or text_features must be provided.")


    def set_mixture_features(self, mixture_features=None):
        if mixture_features is not None:
            # Assume pre-computed features are already normalized if passed directly
            self.mixture_features = mixture_features
        else:
            if self.text_features is None or self.image_features is None:
                 raise ValueError("Image and Text features must be set before calling set_mixture_features without arguments.")
            # Calculate using cached features
            mix_feat = self.mixture_factor_img * self.image_features + self.mixture_factor_text * self.text_features
            self.mixture_features = mix_feat / mix_feat.norm(dim=-1, keepdim=True)


    def set_slerp_features(self, slerp_features=None, alpha=None, static_slerp=False, delta=None, epsilon=None):
        if slerp_features is not None:
            # Assume pre-computed features are already normalized if passed directly
            self.slerp_features = slerp_features
        else:
            if self.text_features is None or self.image_features is None:
                 raise ValueError("Image and Text features must be set before calling set_slerp_features without arguments.")
            # Calculate using cached features
            self.slerp_features = self.get_slerp_features(query_image_features=self.image_features,
                                                          query_text_features=self.text_features,
                                                          alpha=alpha, static_slerp=static_slerp, delta=delta, epsilon=epsilon)

    def set_composed_features(self, composed_features=None):
        raise NotImplementedError("ZeroShotBLIP2 does not have a 'combiner' module integrated.")

    @torch.no_grad()
    def reset_all(self):
        # reset the state dict of clip model, sometimes you may change the weights
        self.clip_model.load_state_dict(self.clip_state_dict)
        self.initial_state_dict = copy.deepcopy(self.clip_model.state_dict())
        if self.momentum_update:
            self.momentum_state_dict = copy.deepcopy(self.clip_model.state_dict())

    @torch.no_grad()
    def reset_initial(self):
        self.clip_model.load_state_dict(self.initial_state_dict)

    @torch.no_grad()
    def momentum_update_model(self):
        update_w = self.update_w
        if self.momentum_update:
            self.update_counter += 1
            # reload momentum state_dict
            state_dict = self.clip_model.state_dict()
            for k, v in state_dict.items():
                self.momentum_state_dict[k]= self.momentum * self.momentum_state_dict[k] + (1.0 - self.momentum) * v

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                for k, v in state_dict.items():
                    # self.initial_state_dict[k] = self.momentum_state_dict[k]
                    self.initial_state_dict[k] = (1 - update_w) * self.clip_state_dict[k] + update_w * self.momentum_state_dict[k]
            # update will be done by function self.reset_initial()



class ZeroShotALBEF(nn.Module):
    def __init__(self, cfg, model, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        self.visual_encoder = model.visual_encoder
        self.text_encoder = model.text_encoder
        self.vision_proj = model.vision_proj
        self.text_proj = model.text_proj
        self.tokenize = model.tokenizer

        self.mixture_factor_img = cfg.OPTIM.MIXTURE_FACTOR_IMG
        self.mixture_factor_text = cfg.OPTIM.MIXTURE_FACTOR_TEXT
        self.use_static_slerp = cfg.OPTIM.USE_STATIC_SLERP
        self.delta = cfg.OPTIM.DELTA
        self.epsilon = cfg.OPTIM.EPSILON
        self.fusion_type = cfg.OPTIM.FUSION_TYPE
        
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.logit_scale = torch.tensor(100.0)
        self.max_txt_len = 30
        
        # For feature caching and dynamic slerp
        self.image_features = None
        self.text_features = None
        self.mixture_features = None
        self.slerp_features = None
        self.composed_features = None
        self.target_image_features = None # Initialized if get_target_image_features_pool is called


    def encode_image(self, image):
        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
        image_features = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        
        return image_features

    def encode_text(self, text):
        text = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_output = self.text_encoder.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
        text_embeds = text_output.last_hidden_state[:, 0, :]
        text_features = F.normalize(self.text_proj(text_embeds), dim=-1)
 
        return text_features


    def extract_features(self, samples, mode="multimodal"):
        image = samples.get("image")
        text = samples.get("text_input")
        
        image_features, text_features, multimodal_features = None, None, None
        
        if mode == "image":
            assert image is not None, "Image is required for image mode"
            image_embeds = self.visual_encoder.forward_features(image)
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
            
        elif mode == "text":
            assert text is not None, "Text is required for text mode"
            text = self.tokenize(
                text,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            text_output = self.text_encoder.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state
            text_features = F.normalize(self.text_proj(text_embeds), dim=-1)
            
        elif mode == "multimodal":
            assert image is not None and text is not None, "Both image and text required for multimodal mode"
            
            image_embeds = self.visual_encoder.forward_features(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
            text = self.tokenize(
                text,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            text_output = self.text_encoder.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state
            
            output = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode="fusion",
            )
            multimodal_features = output.last_hidden_state

        return {
            "image_features": image_features,
            "text_features": text_features,
            "multimodal_features": multimodal_features
        }
    
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
    

    def nlerp_mixture(self, v0, v1, alpha=0.8, epsilon=1.0, static_nlerp=False, dis_type="frobenius"):
        if static_nlerp:
            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        else:
            if dis_type == "cosine":
                distance = F.cosine_similarity(v0, v1, dim=-1, eps=1e-8).unsqueeze(-1)
            elif dis_type == "l2":
                distance = torch.norm(v1 - v0, p=2, dim=-1, keepdim=True)
            elif dis_type == "frobenius":
                distance = torch.norm(v1 - v0, p='fro', dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid distance type: {dis_type}")

            alpha = torch.tanh(epsilon * distance)
            alpha = torch.clamp(alpha, min=0.79, max=0.83)

            nlerp_features = (1-alpha) * v0 + alpha * v1
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)
        return nlerp_features


    def forward(self, images=None, text=None, target_images=None, need_target=False, need_all_logits=False):     
        query_image_features = self.encode_image(images) if images is not None else self.image_features
        query_text_features = self.encode_text(text) if text is not None else self.text_features
        target_image_features = self.get_target_image_features_pool(target_images) if target_images is not None else self.target_image_features

        mixture_features = 0.5 * query_image_features + 0.5 * query_text_features
        mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)

        lerp_features = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features
        lerp_features = lerp_features / lerp_features.norm(dim=-1, keepdim=True)

        nlerp_features = self.nlerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_nlerp=self.use_static_slerp, epsilon=self.epsilon)

        slerp_features = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=self.use_static_slerp, delta=self.delta, epsilon=self.epsilon)

        slerp_features_static = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=True)

        composed_features = lerp_features

        logit_scale = self.logit_scale

        if self.fusion_type == "slerp":
            logits_per_query = logit_scale * slerp_features @ target_image_features.t()
        elif self.fusion_type == "avg":
            logits_per_query = logit_scale * mixture_features @ target_image_features.t()
        elif self.fusion_type == "lerp":
            logits_per_query = logit_scale * lerp_features @ target_image_features.t()


        if need_target and not need_all_logits:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query 
        elif need_target and need_all_logits:
            logits_query_image = logit_scale * query_image_features @ target_image_features.t()
            logits_query_text = logit_scale * query_text_features @ target_image_features.t()
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query, logits_query_image, logits_query_text
        else:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, logits_per_query 


    def forward_with_perturb(self, images=None, text=None, target_images=None, need_target=False, need_all_logits=False):     
        query_image_features = self.encode_image(images) if images is not None else self.image_features
        query_text_features = self.encode_text(text) if text is not None else self.text_features
        target_image_features = self.get_target_image_features_pool(target_images) if target_images is not None else self.target_image_features

        mixture_features = 0.5 * query_image_features + 0.5 * query_text_features
        mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)

        lerp_features = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features
        lerp_features = lerp_features / lerp_features.norm(dim=-1, keepdim=True)

        nlerp_features = self.nlerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_nlerp=self.use_static_slerp, epsilon=self.epsilon)

        slerp_features = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=self.use_static_slerp, delta=self.delta, epsilon=self.epsilon)

        slerp_features_static = self.slerp_mixture(query_image_features, query_text_features, alpha=self.mixture_factor_text, static_slerp=True)

        composed_features = lerp_features

        new_text = self.get_parsing_for_text(text)
        query_text_features_counterfactual= self.encode_text(new_text)
        lerp_features_counterfactual = (1-self.mixture_factor_text) * query_image_features + self.mixture_factor_text * query_text_features_counterfactual
        lerp_features_counterfactual = lerp_features_counterfactual / lerp_features_counterfactual.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale

        if self.fusion_type == "slerp":
            logits_per_query = logit_scale * slerp_features @ target_image_features.t()
        elif self.fusion_type == "avg":
            logits_per_query = logit_scale * mixture_features @ target_image_features.t()
        elif self.fusion_type == "lerp":
            logits_per_query = logit_scale * lerp_features @ target_image_features.t()

        
        if need_target and not need_all_logits:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query 
        elif need_target and need_all_logits:
            logits_query_image = logit_scale * query_image_features @ target_image_features.t()
            logits_query_text = logit_scale * query_text_features @ target_image_features.t()
            logits_query_text_counterfactual = logit_scale * lerp_features_counterfactual @ target_image_features.t()
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, target_image_features, logits_per_query, logits_query_image, logits_query_text, logits_query_text_counterfactual
        else:
            return query_image_features, query_text_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, logits_per_query 

    def get_parsing_for_text(self, text):
        if isinstance(text, tuple) or isinstance(text, list):
            text = text[0]
        
        if not isinstance(text, str):
            raise ValueError(f"Expected string or tuple of strings, but got {type(text)}")
        
        nlp = en_core_web_sm.load()
        doc = nlp(text)
        nouns = [w for w in doc if w.pos_ == "NOUN"]
        verbs = [w for w in doc if w.pos_ == "VERB"]
        adjs = [w for w in doc if w.pos_ == "ADJ"]
        advs = [w for w in doc if w.pos_ == "ADV"]

        parsing = nouns + verbs + adjs + advs

        if not parsing:
            return text

        mask_more_words = True
        if mask_more_words: 
            new_text = text
            for word in parsing:
                new_text = new_text.replace(word.text, '[MASK]')
        else:
            random_word = random.choice(parsing)
            new_text = text.replace(random_word.text, '[MASK]')
        return new_text

    
    def get_text_features(self, text=None, tokenized_prompts=None):
        if tokenized_prompts is None:
            text_features = self.encode_text(text)
        else:
            text_features = self.encode_text(tokenized_prompts)
        return text_features

    def get_image_features(self, images, Global_only=True):
        # ALBEF's encode_image provides global features. Global_only is kept for API consistency.
        image_features = self.encode_image(images)
        return image_features

    def get_target_image_features_pool(self, target_images):
        # Assuming target_images are already processed features
        target_image_features = target_images
        self.target_image_features = target_image_features # Cache it
        return target_image_features

    def get_mixture_features(self, text, image):
        query_text_features = self.encode_text(text)
        query_image_features = self.encode_image(image)
        
        mixture_features = self.mixture_factor_img * query_image_features + self.mixture_factor_text * query_text_features
        mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
        return mixture_features

    def get_slerp_features(self, query_image_features, query_text_features, alpha=None, static_slerp=None, delta=None, epsilon=None):
        img_feat = query_image_features
        txt_feat = query_text_features
        
        if img_feat is None or txt_feat is None:
             raise ValueError("Image and Text features must be provided to get_slerp_features.")

        alpha_val = alpha if alpha is not None else self.mixture_factor_text
        static_slerp_val = static_slerp if static_slerp is not None else self.use_static_slerp
        delta_val = delta if delta is not None else self.delta
        epsilon_val = epsilon if epsilon is not None else self.epsilon

        slerp_features = self.slerp_mixture(img_feat, txt_feat, alpha=alpha_val, static_slerp=static_slerp_val, delta=delta_val, epsilon=epsilon_val)
        return slerp_features # slerp_mixture already normalizes

    def get_composed_features(self, query_image_features, query_text_features):
        if self.combiner is None:
            raise AttributeError("self.combiner is not set for ZeroShotALBEF. Cannot compute composed features.")
        composed_features = self.combiner.combine_features(query_image_features, query_text_features)
        # The combiner in CLIPRet_TTA does not normalize output, so normalize here.
        composed_features = composed_features / composed_features.norm(dim=-1, keepdim=True)
        return composed_features

    def set_image_features(self, images=None, image_features=None, Global_only=True):
        if images is not None:
            self.image_features = self.encode_image(images) # encode_image should normalize
        elif image_features is not None:
            self.image_features = image_features # Assume pre-normalized
        else:
            raise ValueError("Either images or image_features must be provided to set_image_features.")

    def set_text_features(self, text=None, text_features=None):
        if text is not None:
            self.text_features = self.encode_text(text) # encode_text should normalize
        elif text_features is not None:
            self.text_features = text_features # Assume pre-normalized
        else:
            raise ValueError("Either text or text_features must be provided to set_text_features.")

    def set_mixture_features(self, mixture_features=None):
        if mixture_features is not None:
            self.mixture_features = mixture_features # Assume pre-normalized
        else:
            if self.image_features is None or self.text_features is None:
                raise ValueError("self.image_features and self.text_features must be set before calling set_mixture_features without arguments.")
            mix_feat = self.mixture_factor_img * self.image_features + self.mixture_factor_text * self.text_features
            self.mixture_features = mix_feat / mix_feat.norm(dim=-1, keepdim=True)

    def set_slerp_features(self, slerp_features=None, alpha=None, static_slerp=None, delta=None, epsilon=None):
        if slerp_features is not None:
            self.slerp_features = slerp_features # Assume pre-normalized
        else:
            if self.image_features is None or self.text_features is None:
                raise ValueError("self.image_features and self.text_features must be set before calling set_slerp_features without arguments.")
            self.slerp_features = self.get_slerp_features(
                query_image_features=self.image_features,
                query_text_features=self.text_features,
                alpha=alpha, 
                static_slerp=static_slerp,
                delta=delta,
                epsilon=epsilon
            )

    def set_composed_features(self, composed_features=None):
        if composed_features is not None:
            self.composed_features = composed_features # Assume pre-normalized
        else:
            if self.combiner is None:
                raise AttributeError("self.combiner is not set. Cannot compute composed_features.")
            if self.image_features is None or self.text_features is None:
                raise ValueError("self.image_features and self.text_features must be set before calling set_composed_features without arguments.")
            self.composed_features = self.get_composed_features(
                query_image_features=self.image_features,
                query_text_features=self.text_features
            )


def get_model(cfg, num_classes: int, device: Union[str, torch.device]):
    """
    Setup the pre-defined model architecture and restore the corresponding pre-trained weights
    Input:
        cfg: Configurations
        num_classes: Number of classes
        device: The device to put the loaded model
    Return:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    preprocess = None

    if cfg.MODEL.USE_CLIP:
        # load pre-trained CLIP model
        base_model, _, preprocess = create_model_and_transforms(cfg.MODEL.ARCH, pretrained=cfg.MODEL.WEIGHTS, device=device, precision=cfg.CLIP.PRECISION)

        """
        'ViT-B-32':'laion2b_s34b_b79k', 'ViT-B-16':'laion2b_s34b_b88k',
        'ViT-L-14':'laion2b_s32b_b82k', 'ViT-H-14':'laion2b_s32b_b79k',
        'ViT-g-14':'laion2b_s34b_b88k', 'ViT-bigG-14':'laion2b_s39b_b160k'
        """
        # get the image input normalization
        normalization = preprocess.transforms[-1]
        # remove the input normalization from the pre-processing as it will be added to the model
        preprocess.transforms = preprocess.transforms[:-1]

        if cfg.MODEL.ADAPTATION == "tpt":
            base_model = ClipTestTimePromptTuning(base_model, normalization,
                                                  cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET,
                                                  n_ctx=cfg.TPT.N_CTX, ctx_init=cfg.TPT.CTX_INIT,
                                                  class_token_pos=cfg.TPT.CLASS_TOKEN_POS)
            if cfg.MODEL.CKPT_PATH:
                # Initiaize context prompts with CoOp pre-trained prompts (see: https://github.com/KaiyangZhou/CoOp?tab=readme-ov-file)
                # or download them from here: https://drive.google.com/file/d/18ypxfd82RR0pizc5MM1ZWDYDk4j0BtPF/view
                pretrained_ctx = torch.load(cfg.MODEL.CKPT_PATH)['state_dict']['ctx']
                assert pretrained_ctx.shape[0] == cfg.TPT.N_CTX
                with torch.no_grad():
                    base_model.prompt_learner.ctx.copy_(pretrained_ctx)
                    base_model.prompt_learner.ctx_init_state = pretrained_ctx
                logger.info("Successfully restored pre-trained soft prompt (CoOp)")
        else: 
            # img2text = IM2TEXT(embed_dim=512, middle_dim=512, output_dim=base_model.token_embedding.weight.shape[1], n_layer=2)
            if cfg.MODEL.ARCH == "ViT-B-16" or cfg.MODEL.ARCH == "ViT-B-32":
                img2text = IM2TEXT(embed_dim=512, middle_dim=512, output_dim=512, n_layer=2)
                combiner = Combiner(clip_feature_dim=512, projection_dim=512, hidden_dim=512)
            elif cfg.MODEL.ARCH == "ViT-L-14":
                img2text = IM2TEXT(embed_dim=768, middle_dim=768, output_dim=768, n_layer=2)
                combiner = Combiner(clip_feature_dim=768, projection_dim=768, hidden_dim=768)
            
            if cfg.MODEL.ADAPTATION == "source":
                base_model = ZeroShotCLIP(cfg, base_model, device, normalize=normalization)  
            else:
                base_model = CLIPRet_TTA(cfg, base_model, combiner, device)
            convert_models_to_fp32(base_model)
            # convert_models_to_fp16(base_model)
    
    elif cfg.MODEL.USE_BLIP:
        # model, vis_processors, _ = lavis_load_model_and_preprocess(name="Blip2Base", model_type="base", is_eval=True, device=device)
        base_model, vis_processors, _ = lavis_load_model_and_preprocess(name="blip2", model_type="pretrain", is_eval=False, device=device)

        base_model = ZeroShotBLIP2(cfg, base_model, device)
        img2text = IM2TEXT(embed_dim=512, middle_dim=512, output_dim=512, n_layer=2)
        combiner = Combiner(clip_feature_dim=512, projection_dim=512, hidden_dim=512)
        convert_models_to_fp32(base_model)
    
    elif cfg.MODEL.USE_ALBEF:
        base_model, vis_processors, _ = lavis_load_model_and_preprocess(name="albef_feature_extractor", model_type="base", is_eval=False, device=device)
        base_model = ZeroShotALBEF(cfg, base_model, device)
        img2text = IM2TEXT(embed_dim=256, middle_dim=256, output_dim=256, n_layer=2)
        combiner = Combiner(clip_feature_dim=256, projection_dim=256, hidden_dim=256)
        convert_models_to_fp32(base_model)

    else:  # not use CLIP model
        try:
            # load model from torchvision
            base_model, preprocess = get_torchvision_model(cfg.MODEL.ARCH, weight_version=cfg.MODEL.WEIGHTS)
        except ValueError:
            try:
                # load model from timm
                base_model, preprocess = get_timm_model(cfg.MODEL.ARCH)
            except ValueError:
                pass

    return base_model.to(device), img2text.to(device), combiner.to(device), preprocess

