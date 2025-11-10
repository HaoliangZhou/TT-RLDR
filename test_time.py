import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import logging
import numpy as np
import methods
import json
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from PIL import Image
from models.model import get_model
from models.model_reward import get_reward_model
from utils.misc import print_memory_info
from utils.eval_utils import evaluate_fashion, evaluate_cirr, evaluate_cirr_test, evaluate_coco
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')

def _transform(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

def evaluate(description):
    load_cfg_from_args(description)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, img2text, combiner, model_preprocess = get_model(cfg, num_classes, device)

    model_preprocess = _transform(224, is_train=False)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"

    if cfg.CORRUPTION.DATASET in ["fashioniq", "cirr", "cirr_test", "coco"]:
        if cfg.MODEL.ADAPTATION == "reward" or cfg.MODEL.ADAPTATION == "kd":
            reward_model = get_reward_model(cfg, device)
            model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, reward_model=reward_model, combiner=combiner, num_classes=num_classes)
            logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}, with reward model: {cfg.REWARD.REWARD_ARCH}")
        else:
            model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, combiner=combiner, num_classes=num_classes)
            logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # start evaluation
    model.reset()
    logger.info("resetting model")
    domain_name = ''
    domain_sequence = ''
    severity = ''
        
    test_source_dataset, test_target_dataset = get_test_loader(
        setting=cfg.SETTING,
        adaptation=cfg.MODEL.ADAPTATION,
        dataset_name=cfg.CORRUPTION.DATASET,
        dataset_fiqtype=cfg.CORRUPTION.DATASET_FIQTYPE,
        preprocess=model_preprocess,
        data_root_dir=cfg.DATA_DIR,
        domain_name=domain_name,
        domain_names_all=domain_sequence,
        severity=severity,
        num_examples=cfg.CORRUPTION.NUM_EX,
        rng_seed=cfg.RNG_SEED,
        use_clip=cfg.MODEL.USE_CLIP,
        n_views=cfg.TEST.N_AUGMENTATIONS,
        delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
        batch_size=cfg.TEST.BATCH_SIZE,
        batch_size_target=cfg.TEST.BATCH_SIZE_TARGET,
        shuffle=False,
        workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
    )

    # Note that the input normalization is done inside of the model
    logger.info(f"Using the following data transformation:\n{test_source_dataset.dataset.transforms}")


    # evaluate the model
    if cfg.CORRUPTION.DATASET == "fashioniq":
        if cfg.MODEL.ADAPTATION == "reward" or cfg.MODEL.ADAPTATION == "kd":
            evaluate_fashion(model=model, reward_model=reward_model, img2text=img2text, cfg=cfg, source_loader=test_source_dataset, target_loader=test_target_dataset, device=device)
        else:
            evaluate_fashion(model=model, img2text=img2text, cfg=cfg, source_loader=test_source_dataset, target_loader=test_target_dataset, device=device)
    
    elif cfg.CORRUPTION.DATASET == "cirr":
        if cfg.MODEL.ADAPTATION == "reward" or cfg.MODEL.ADAPTATION == "kd":
            _, feats, target_image_features = evaluate_cirr(model=model, reward_model=reward_model, img2text=img2text, cfg=cfg, query_loader=test_source_dataset, target_loader=test_target_dataset, device=device)
        else:
            _, feats, target_image_features = evaluate_cirr(model=model, img2text=img2text, cfg=cfg, query_loader=test_source_dataset, target_loader=test_target_dataset, device=device)
    
    elif cfg.CORRUPTION.DATASET == "cirr_test":
        if cfg.MODEL.ADAPTATION == "reward" or cfg.MODEL.ADAPTATION == "kd":
            results=evaluate_cirr_test(model=model, reward_model=reward_model, img2text=img2text, cfg=cfg, query_loader=test_source_dataset, target_loader=test_target_dataset, device=device)
        else:
            results=evaluate_cirr_test(model=model, img2text=img2text, cfg=cfg, query_loader=test_source_dataset, target_loader=test_target_dataset, device=device)
        
        for key, value in results.items():
            with open(cfg.SAVE_DIR + '/reward' + '_' + key + '.json', 'w') as f:
                json.dump(value, f)            
    
    elif cfg.CORRUPTION.DATASET == "coco":
        if cfg.MODEL.ADAPTATION == "reward" or cfg.MODEL.ADAPTATION == "kd":
            evaluate_coco(model=model, reward_model=reward_model, img2text=img2text, cfg=cfg, loader=test_source_dataset, device=device)
        else:
            evaluate_coco(model=model, img2text=img2text, cfg=cfg, loader=test_source_dataset, device=device)
    


    logger.info(f"{cfg.CORRUPTION.DATASET} retrieval done! Any Details: {cfg.DESC}")

    if cfg.TEST.DEBUG:
        print_memory_info()


if __name__ == '__main__':
    evaluate('"Evaluation.')
