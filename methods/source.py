
# from methods.base import TTAMethod, forward_decorator

from methods.base import TTAMethod
from copy import deepcopy
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class Source(TTAMethod):
    def __init__(self, cfg, model, combiner, num_classes):
        super().__init__(cfg, model, combiner, num_classes)

        self.cfg = cfg

    # @forward_decorator
    def forward_and_adapt(self, x, y, z):
        if self.cfg.MODEL.USE_CLIP:
            query_image_features = self.model.encode_image(x)
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)

            target_image_features = self.model.encode_image(z)
            target_image_features = target_image_features / target_image_features.norm(dim=-1, keepdim=True)

            caption_features = self.model.encode_text(y)
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)

            mixture_features = 0.5 * query_image_features + 0.5 * caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)   

            lerp_features = (1-self.cfg.OPTIM.MIXTURE_FACTOR_TEXT) * query_image_features + self.cfg.OPTIM.MIXTURE_FACTOR_TEXT * caption_features

            nlerp_features = self.model.nlerp_mixture(query_image_features, caption_features, alpha=self.cfg.OPTIM.MIXTURE_FACTOR_TEXT)

            slerp_features = self.model.slerp_mixture(query_image_features, caption_features, alpha=self.cfg.OPTIM.MIXTURE_FACTOR_TEXT, static_slerp=self.cfg.OPTIM.USE_STATIC_SLERP, delta=self.cfg.OPTIM.DELTA, epsilon=self.cfg.OPTIM.EPSILON)

            slerp_features_static = self.model.slerp_mixture(query_image_features, caption_features, alpha=self.cfg.OPTIM.MIXTURE_FACTOR_TEXT, static_slerp=True)

        else:
            query_image_features = self.model.encode_image(x)
            target_image_features = self.model.encode_image(z)
            caption_features = self.model.encode_text(y)

            mixture_features = 0.5 * query_image_features + 0.5 * caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)   

            lerp_features = (1-self.cfg.OPTIM.MIXTURE_FACTOR_TEXT) * query_image_features + self.cfg.OPTIM.MIXTURE_FACTOR_TEXT * caption_features
            lerp_features = lerp_features / lerp_features.norm(dim=-1, keepdim=True)

            nlerp_features = self.model.nlerp_mixture(query_image_features, caption_features, alpha=self.cfg.OPTIM.MIXTURE_FACTOR_TEXT)
            nlerp_features = nlerp_features / nlerp_features.norm(dim=-1, keepdim=True)

            slerp_features = self.model.slerp_mixture(query_image_features, caption_features, alpha=self.cfg.OPTIM.MIXTURE_FACTOR_TEXT, static_slerp=self.cfg.OPTIM.USE_STATIC_SLERP, delta=self.cfg.OPTIM.DELTA, epsilon=self.cfg.OPTIM.EPSILON)
            slerp_features = slerp_features / slerp_features.norm(dim=-1, keepdim=True)

            slerp_features_static = self.model.slerp_mixture(query_image_features, caption_features, alpha=self.cfg.OPTIM.MIXTURE_FACTOR_TEXT, static_slerp=True)
            slerp_features_static = slerp_features_static / slerp_features_static.norm(dim=-1, keepdim=True)

            composed_features = lerp_features
            
        
        return query_image_features, caption_features, target_image_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = None
        return model_states, optimizer_state

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

    def encode_text(self, text):
        # return self.model.encode_text(text)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image):
        # return self.model.encode_image(image)
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


