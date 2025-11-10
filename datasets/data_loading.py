import os
import logging
import random
import numpy as np
import time
import webdataset as wds

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from typing import Union
from conf import complete_data_dir_path
from datasets.data_coir import FashionIQ, CIRR, CsvCOCO
from augmentations.transforms_adacontrast import get_augmentation_versions, get_augmentation
from augmentations.transforms_augmix import AugMixAugmenter


logger = logging.getLogger(__name__)


def identity(x):
    return x


def get_transform(dataset_name: str, adaptation: str, preprocess: Union[transforms.Compose, None], use_clip: bool, n_views: int = 64):
    """
    Get the transformation pipeline
    Note that the data normalization is done within the model
    Input:
        dataset_name: Name of the dataset
        adaptation: Name of the adaptation method
        preprocess: Input pre-processing from restored model (if available)
        use_clip: If the underlying model is based on CLIP
        n_views Number of views for test-time augmentation
    Returns:
        transforms: The data pre-processing (and augmentation)
    """
    if use_clip:
        transform = preprocess
    else:
        if preprocess:
            # set transform to the corresponding input transformation of the restored model
            transform = preprocess
        else:
            # use classical ImageNet transformation procedure
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

    return transform


def get_test_loader(setting: None, adaptation: str, dataset_name: None, dataset_fiqtype: None, preprocess: Union[transforms.Compose, None],
                    data_root_dir: str, domain_name: None, domain_names_all: None, severity: None, num_examples: int,
                    rng_seed: int, use_clip: bool, n_views: int = 64, delta_dirichlet: float = 0.,
                    batch_size: int = 128, batch_size_target: int = 128, shuffle: bool = False, workers: int = 4, cfg=None):
    """
    Create the test data loader
    Input:
        setting: Name of the considered setting
        adaptation: Name of the adaptation method
        dataset_name: Name of the dataset
        preprocess: Input pre-processing from restored model (if available)
        data_root_dir: Path of the data root directory
        domain_name: Name of the current domain
        domain_names_all: List containing all domains
        severity: Severity level in case of corrupted data
        num_examples: Number of test samples for the current domain
        rng_seed: A seed number
        use_clip: If the underlying model is based on CLIP
        n_views: Number of views for test-time augmentation
        delta_dirichlet: Parameter of the Dirichlet distribution
        batch_size: The number of samples to process in each iteration
        shuffle: Whether to shuffle the data. Will destroy pre-defined settings
        workers: Number of workers used for data loading
    Returns:
        test_loader: The test data loader
    """

    # Fix seed again to ensure that the test sequence is the same for all methods
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    data_dir = complete_data_dir_path(data_root_dir, dataset_name)
    logger.info("load data from {}".format(data_dir))
    transform = get_transform(dataset_name, adaptation, preprocess, use_clip, n_views)

    # create the test dataset
    if dataset_name == "fashioniq":
        source_data = dataset_fiqtype
        logger.info(f"load FashionIQ: {source_data} as tta dataset")
        assert source_data in ['dress', 'toptee', 'shirt']
        test_query_dataset = FashionIQ(cloth=source_data,
                                    transforms=transform,
                                    root=data_dir,
                                    is_return_target_path=True,
                                    if_clip=use_clip)
        test_target_dataset = FashionIQ(cloth=source_data,
                                    transforms=transform,
                                    root=data_dir,
                                    mode='imgs',
                                    if_clip=use_clip)
    elif dataset_name == "cirr":
        test_query_dataset = CIRR(transforms=transform, root=data_dir, if_clip=use_clip)
        test_target_dataset = CIRR(transforms=transform, mode='imgs', if_clip=use_clip)
    elif dataset_name == "cirr_test":
        test_query_dataset = CIRR(transforms=transform, root=data_dir, if_clip=use_clip, test=True)
        test_target_dataset = CIRR(transforms=transform, mode='imgs', root=data_dir, if_clip=use_clip, test=True)
    elif dataset_name == "coco":
        trans_val = preprocess.transforms
        n_px = trans_val[1].size
        trans_val = [transforms.Resize(n_px, interpolation=Image.BICUBIC)] + trans_val[2:]
        preprocess_val_region = transforms.Compose(trans_val)
        test_query_dataset = CsvCOCO(transforms=transform, transforms_region=preprocess_val_region, root=data_dir)
        test_target_dataset = CsvCOCO(transforms=transform, transforms_region=preprocess_val_region, root=data_dir) 
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    try:
        # shuffle the test sequence; deterministic behavior for a fixed random seed
        random.shuffle(test_query_dataset.samples)
        random.shuffle(test_target_dataset.samples)

        # randomly subsample the dataset if num_examples is specified
        if num_examples != -1:
            num_samples_orig = len(test_query_dataset)
            # logger.info(f"Changing the number of test samples from {num_samples_orig} to {num_examples}...")
            test_query_dataset.samples = random.sample(test_query_dataset.samples, k=min(num_examples, num_samples_orig))
    except AttributeError:
        logger.warning("Attribute 'samples' is missing. Continuing without shuffling, sorting or subsampling the files...")

    return (torch.utils.data.DataLoader(test_query_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False),
            torch.utils.data.DataLoader(test_target_dataset, batch_size=batch_size_target, shuffle=shuffle, num_workers=workers, drop_last=False),)



def sort_by_dirichlet(delta_dirichlet: float, samples: list):
    """
    Adapted from: https://github.com/TaesikGong/NOTE/blob/main/learner/dnn.py
    Sort classes according to a dirichlet distribution
    Input:
        delta_dirichlet: Parameter of the distribution
        samples: List containing all data sample pairs (file_path, class_label)
    Returns:
        samples_sorted: List containing the temporally correlated samples
    """

    N = len(samples)
    samples_sorted = []
    class_labels = np.array([val[1] for val in samples])
    num_classes = int(np.max(class_labels) + 1)
    dirichlet_numchunks = num_classes

    time_start = time.time()
    time_duration = 120  # seconds until program terminates if no solution was found

    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
    min_size = -1
    min_size_thresh = 10
    while min_size < min_size_thresh:  # prevent any chunk having too less data
        idx_batch = [[] for _ in range(dirichlet_numchunks)]
        idx_batch_cls = [[] for _ in range(dirichlet_numchunks)]  # contains data per each class
        for k in range(num_classes):
            idx_k = np.where(class_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(delta_dirichlet, dirichlet_numchunks))

            # balance
            proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

            # store class-wise data
            for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                idx_j.append(idx)

        # exit loop if no solution was found after a certain while
        if time.time() > time_start + time_duration:
            raise ValueError(f"Could not correlated sequence using dirichlet value '{delta_dirichlet}'. Try other value!")

    sequence_stats = []

    # create temporally correlated sequence
    for chunk in idx_batch_cls:
        cls_seq = list(range(num_classes))
        np.random.shuffle(cls_seq)
        for cls in cls_seq:
            idx = chunk[cls]
            samples_sorted.extend([samples[i] for i in idx])
            sequence_stats.extend(list(np.repeat(cls, len(idx))))

    return samples_sorted
