# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

_C.SETTING = "reset_each_shift"

# Data directory
_C.DATA_DIR = ""

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Output directory
_C.SAVE_DIR = "./output/"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Enables printing intermediate results every x batches.
# Default -1 corresponds to no intermediate results
_C.PRINT_EVERY = -1

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 1

# Deterministic experiments.
_C.DETERMINISM = False

# Precision
_C.MIXED_PRECISION = True

# Optional description of a config
_C.DESC = ""

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Some of the available models can be found here:
# Torchvision: https://pytorch.org/vision/0.14/models.html
# timm: https://github.com/huggingface/pytorch-image-models/tree/v0.6.13
# OpenCLIP: https://github.com/mlfoundations/open_clip
_C.MODEL.ARCH = 'Standard'

# Type of pre-trained weights
# For torchvision models see: https://pytorch.org/vision/0.14/models.html
# For OpenClip models, use either 'openai' (for the original OpenAI weights) or see https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
_C.MODEL.WEIGHTS = "IMAGENET1K_V1"

# Whether to use a CLIP based architecture
_C.MODEL.USE_CLIP = False

# Whether to use a BLIP based architecture
_C.MODEL.USE_BLIP = False

# Whether to use a ALBEF based architecture
_C.MODEL.USE_ALBEF = False

# Path to a specific checkpoint
_C.MODEL.CKPT_PATH = ""

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

# Reset the model after a certain amount of update steps (e.g., used in RDumb)
_C.MODEL.RESET_AFTER_NUM_UPDATES = 0

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cirr'

_C.CORRUPTION.DATASET_FIQTYPE= 'toptee'  # ['dress', 'toptee', 'shirt']

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate. If num_ex != -1, each sequence is sub-sampled to the specified amount
# For ImageNet-C, RobustBench loads a list containing 5000 samples.
_C.CORRUPTION.NUM_EX = -1

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN alpha (1-alpha) * src_stats + alpha * test_stats
_C.BN.ALPHA = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Optimizer choices: Adam, AdamW, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta1 for Adam based optimizers
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# Mixture factor for image and text features
_C.OPTIM.MIXTURE_FACTOR_IMG = 0.5
_C.OPTIM.MIXTURE_FACTOR_TEXT = 0.5

# Whether to use static slerp
_C.OPTIM.FUSION_TYPE = "lerp"
_C.OPTIM.USE_STATIC_SLERP = False
_C.OPTIM.DELTA = 0.1
_C.OPTIM.EPSILON = 1.0

# --------------------------------- Mean teacher options -------------------- #
_C.M_TEACHER = CfgNode()

# Mean teacher momentum for EMA update
_C.M_TEACHER.MOMENTUM = 0.999

# --------------------------------- Contrastive options --------------------- #
_C.CONTRAST = CfgNode()

# Temperature term for contrastive learning
_C.CONTRAST.TEMPERATURE = 0.1

# Output dimension of projector
_C.CONTRAST.PROJECTION_DIM = 128

# Contrastive mode
_C.CONTRAST.MODE = 'all'

# --------------------------------- CoTTA options --------------------------- #
_C.COTTA = CfgNode()

# Restore probability
_C.COTTA.RST = 0.01

# Average probability for TTA
_C.COTTA.AP = 0.92

# --------------------------------- GTTA options ---------------------------- #
_C.GTTA = CfgNode()

_C.GTTA.STEPS_ADAIN = 1
_C.GTTA.PRETRAIN_STEPS_ADAIN = 20000
_C.GTTA.LAMBDA_MIXUP = 1/3
_C.GTTA.USE_STYLE_TRANSFER = False

# --------------------------------- RMT options ----------------------------- #
_C.RMT = CfgNode()

_C.RMT.LAMBDA_CE_SRC = 1.0          # Lambda for source replay. Set to 0 for source-free variant
_C.RMT.LAMBDA_CE_TRG = 1.0          # Lambda for self-training
_C.RMT.LAMBDA_CONT = 1.0            # Lambda for contrastive learning
_C.RMT.NUM_SAMPLES_WARM_UP = 50000  # Number of samples used during the mean teacher warm-up

# --------------------------------- SANTA options --------------------------- #
_C.SANTA = CfgNode()

_C.SANTA.LAMBDA_CE_TRG = 1.0        # Lambda for self-training
_C.SANTA.LAMBDA_CONT = 1.0          # Lambda for contrastive learning

# --------------------------------- AdaContrast options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.QUEUE_SIZE = 16384
_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard" # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0          # Lambda for classification loss
_C.ADACONTRAST.BETA = 1.0           # Lambda for instance loss
_C.ADACONTRAST.ETA = 1.0            # Lambda for diversity loss

_C.ADACONTRAST.DIST_TYPE = "cosine"         # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong"  # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# --------------------------------- LAME options ---------------------------- #
_C.LAME = CfgNode()

_C.LAME.AFFINITY = "rbf"
_C.LAME.KNN = 5
_C.LAME.SIGMA = 1.0
_C.LAME.FORCE_SYMMETRY = False

# --------------------------------- EATA options ---------------------------- #
_C.EATA = CfgNode()

# Fisher alpha. If set to 0.0, EATA becomes ETA and no EWC regularization is used
_C.EATA.FISHER_ALPHA = 2000

# Diversity margin
_C.EATA.D_MARGIN = 0.05
_C.EATA.MARGIN_E0 = 0.4             # Will be multiplied by: EATA.MARGIN_E0 * math.log(num_classes)

# --------------------------------- SAR options ---------------------------- #
_C.SAR = CfgNode()

# Threshold e_m for model recovery scheme
_C.SAR.RESET_CONSTANT_EM = 0.2

# --------------------------------- DeYO options ---------------------------- #
_C.DEYO = CfgNode()

_C.DEYO.REWEIGHT_ENT = True
_C.DEYO.REWEIGHT_PLPD = True
_C.DEYO.PLPD = 0.2
_C.DEYO.MARGIN = 0.5                # Will be multiplied by: DEYO.MARGIN * math.log(num_classes)
_C.DEYO.AUG_TYPE = "patch"          # Choose from: ['occ', 'patch', 'pixel']
_C.DEYO.OCCLUSION_SIZE = 112        # For aug_type occ
_C.DEYO.ROW_START = 56              # For aug_type occ
_C.DEYO.COLUMN_START = 56           # For aug_type occ
_C.DEYO.PATCH_LEN = 4               # For aug_type patch

# --------------------------------- ROTTA options -------------------------- #
_C.ROTTA = CfgNode()

_C.ROTTA.MEMORY_SIZE = 64
_C.ROTTA.UPDATE_FREQUENCY = 64
_C.ROTTA.NU = 0.001
_C.ROTTA.ALPHA = 0.05
_C.ROTTA.LAMBDA_T = 1.0
_C.ROTTA.LAMBDA_U = 1.0

# --------------------------------- RPL options ---------------------------- #
_C.RPL = CfgNode()

# Q value of GCE loss
_C.RPL.Q = 0.8

# --------------------------------- ROID options --------------------------- #
_C.ROID = CfgNode()

_C.ROID.USE_WEIGHTING = True        # Whether to use loss weighting
_C.ROID.USE_PRIOR_CORRECTION = True # Whether to use prior correction
_C.ROID.USE_CONSISTENCY = True      # Whether to use consistency loss
_C.ROID.MOMENTUM_SRC = 0.99         # Momentum for weight ensembling (param * model + (1-param) * model_src)
_C.ROID.MOMENTUM_PROBS = 0.9        # Momentum for diversity weighting
_C.ROID.TEMPERATURE = 1/3           # Temperature for weights

# --------------------------------- CMF options --------------------------- #
_C.CMF = CfgNode()

_C.CMF.ALPHA = 0.99
_C.CMF.GAMMA = 0.99
_C.CMF.Q = 0.005
_C.CMF.TYPE = "lp"

# --------------------------------- Reward options --------------------------- #
_C.REWARD = CfgNode()


_C.REWARD.MOMENTUM_UPDATE = 0
_C.REWARD.UPDATE_FREQ = 64
_C.REWARD.UPDATE_W = 1.0
_C.REWARD.MOMENTUM = 0.9998

_C.REWARD.REWARD_ARCH = "ViT-L-14"
_C.REWARD.REWARD_ARCH_WEIGHTS = "openai"
_C.REWARD.CLIP_SCORE_WEIGHT = 2.5
_C.REWARD.REWARD_AMPLIFY = 0
_C.REWARD.SAMPLE_K = 16
_C.REWARD.REWARD_PROCESS = 1
_C.REWARD.PROCESS_BATCH = 0
_C.REWARD.MULTIPLE_REWARD_MODELS = False
_C.REWARD.BASEINE_SCORE = "ema_mean"
_C.REWARD.CF_REWARD_LAMBDA = 0.1
_C.REWARD.SAMPLING_TYPE = "temperature"
_C.REWARD.TEMPERATURE = 1.0
_C.REWARD.TEACHER_TEXT_FACTOR = 0.8
_C.REWARD.USE_SELF_REWARD = True
_C.REWARD.SELF_REWARD_WEIGHT = 1.0


# ------------------------------- CLIP options ---------------------------- #
_C.CLIP = CfgNode()

_C.CLIP.PROMPT_MODE = "custom"                  # Choose from: custom, ensemble, cupl, all_prompts
_C.CLIP.PROMPT_TEMPLATE = ["a photo of a {}."]  # List of custom prompt templates
_C.CLIP.PROMPT_PATH = "datasets/cupl_prompts/CuPL_ImageNet_prompts.json" # Path to .json file containing CuPL prompts for example
_C.CLIP.PRECISION = "fp16"                      # Precision of the restored weights
_C.CLIP.FREEZE_TEXT_ENCODER = True              # Whether to freeze the text encoder in ZeroShotCLIP
_C.CLIP.FIXED_VISUAL_LAYER = 0
_C.CLIP.FIXED_TEXTUAL_LAYER = 0

# ------------------------------- BLIP options ----------------------------- #
_C.BLIP = CfgNode()


# ------------------------------- TPT options ----------------------------- #
_C.TPT = CfgNode()

_C.TPT.SELECTION_P = 0.1            # Percentile of the most certain prediction
_C.TPT.N_CTX = 4                    # Number of tunable context tokens
_C.TPT.CTX_INIT = "a_photo_of_a"    # Context initialization
_C.TPT.CLASS_TOKEN_POS = "end"      # Position of the class token. Choose from: [end, middle, front]

# ------------------------------- Source options -------------------------- #
_C.SOURCE = CfgNode()

# Number of workers for source data loading
_C.SOURCE.NUM_WORKERS = 4

# Percentage of source samples used
_C.SOURCE.PERCENTAGE = 1.0   # (0, 1] Possibility to reduce the number of source samples

# Possibility to define the number of source samples. The default setting corresponds to all source samples
_C.SOURCE.NUM_SAMPLES = -1

# ------------------------------- Testing options ------------------------- #
_C.TEST = CfgNode()

# Number of workers for test data loading
_C.TEST.NUM_WORKERS = 4

# Batch size for evaluation (and updates)
_C.TEST.BATCH_SIZE = 128
_C.TEST.BATCH_SIZE_TARGET = 128

# If the batch size is 1, a sliding window approach can be applied by setting window length > 1
_C.TEST.WINDOW_LENGTH = 1

# Number of augmentations for methods relying on TTA (test time augmentation)
_C.TEST.N_AUGMENTATIONS = 32

# The value of the Dirichlet distribution used for sorting the class labels.
_C.TEST.DELTA_DIRICHLET = 0.0

# Debuging mode
_C.TEST.DEBUG = False

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL.ADAPTATION}_{cfg.CORRUPTION.DATASET}_{current_time}")
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            # enforce determinism
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(data_root_dir: str, dataset_name: str):
    # map dataset name to data directory name
    mapping = {
               "fashioniq": os.path.join("CIR_task"),
               "cirr": os.path.join("CIR_task"),
               "cirr_test": os.path.join("CIR_task"),
               "coco": os.path.join("CIR_task"),
               }
    assert dataset_name in mapping.keys(),\
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(mapping.keys())}"
    return os.path.join(data_root_dir, mapping[dataset_name])





def get_num_classes(dataset_name: str):
    dataset_name2num_classes = {
                                "fashioniq": 64, 
                                "cirr":64, "cirr_test":64, 
                                "coco": 64, 
                                }
    assert dataset_name in dataset_name2num_classes.keys(), \
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(dataset_name2num_classes.keys())}"
    return dataset_name2num_classes[dataset_name]
