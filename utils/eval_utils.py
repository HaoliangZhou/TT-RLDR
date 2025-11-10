import torch
import logging
import os
import time
import numpy as np
from tqdm import tqdm
from typing import Union
from functools import partial
from copy import deepcopy
import torchvision.transforms.functional as F
import torch.nn.functional as FF
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.third_party.open_clip.tokenizer import tokenize

logger = logging.getLogger(__name__)



def evaluate_fashion(model=None, reward_model=None, img2text=None, cfg=None, source_loader=None, target_loader=None, device=None):
    model.eval()
    if img2text is not None:
        img2text.eval()
    if reward_model is not None:
        reward_model.eval()
    all_target_paths = []  
    all_answer_paths = []  
    all_image_features = []  
    all_query_image_features = []
    all_composed_features = []
    all_caption_features = []
    all_mixture_features = []
    all_lerp_features = []
    all_slerp_features = []
    all_nlerp_features = []
    all_slerp_features_static = []
    all_reference_names = []  
    all_captions = []  
    # m = model.model
    start_time = time.time()

    # target image feature computing
    logging.info("Computing target image features for base model...")
    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch # (bs, 3, 224, 224)
            if device is not None:
                target_images = target_images.cuda(device, non_blocking=True)
            image_features = model.encode_image(target_images)  # (bs, 768)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)
        logging.info(f"All_target_image_features.shape: {len(all_image_features)},{all_image_features[0].shape}")

        if reward_model is not None:
            model.set_image_features(image_features=torch.cat(all_image_features, dim=0))
            logging.info(f"set_image_features for base model done, image_features.shape: {model.model.image_features.shape}")
            
            reward_model.set_image_features_with_dataloder(target_loader)
            logging.info(f"set_image_features_with_dataloder for reward model done, use_time: {time.time() - start_time:.2f}s")


    logging.info("Computing query & target features...")
    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch  # answer_path: target image path
            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)
            if device is not None:
                ref_images = ref_images.cuda(device, non_blocking=True)  # (bs, 3, 224, 224)
                target_images = target_images.cuda(device, non_blocking=True)  # (bs, 3, 224, 224)

            if cfg.MODEL.ADAPTATION == "source":
                query_image_features, caption_features, target_image_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features = model(x=ref_images, y=target_caption, z=target_images)
            else:
                query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features = model.forward_and_adapt(query_image=ref_images, query_text=caption_only, target_images_pool=torch.cat(all_image_features, dim=0))
            
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_mixture_features.append(mixture_features)
            all_lerp_features.append(lerp_features)
            all_nlerp_features.append(nlerp_features)
            all_slerp_features.append(slerp_features)
            all_slerp_features_static.append(slerp_features_static)
            all_composed_features.append(composed_features)


        logging.info(f"All_caption_features.shape: {len(all_caption_features)},{all_caption_features[0].shape}")
        logging.info(f"All_query_image_features.shape: {len(all_query_image_features)},{all_query_image_features[0].shape}")
        logging.info(f"All_mixture_features.shape: {len(all_mixture_features)},{all_mixture_features[0].shape}")
        logging.info(f"All_composed_features.shape: {len(all_composed_features)},{all_composed_features[0].shape}")


        metric_func = partial(get_metrics_fashion,
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths,
                              answer_names=all_answer_paths
                              )

        feats = {'image-only': torch.cat(all_query_image_features),
                'text-only': torch.cat(all_caption_features),
                'mixture': torch.cat(all_mixture_features),
                'lerp': torch.cat(all_lerp_features),
                'nlerp': torch.cat(all_nlerp_features),
                'slerp': torch.cat(all_slerp_features),
                'slerp_static': torch.cat(all_slerp_features_static),
                'composed': torch.cat(all_composed_features)
                }

        for key, value in feats.items():
            metrics = metric_func(ref_features=value, exp_type=key)
            logging.info(f"Eval {key} Feature " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_cirr(model=None, reward_model=None, img2text=None, cfg=None, query_loader=None, target_loader=None, device=None):
    model.eval()
    if img2text is not None:
        img2text.eval()
    if reward_model is not None:
        reward_model.eval()
    all_image_features = []
    all_target_image_features_for_gap = []
    all_query_image_features = []
    all_composed_features = []
    all_mixture_features = []
    all_slerp_features = []
    all_nlerp_features = []
    all_slerp_features_static = []
    all_lerp_features = []
    all_caption_features = []
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_raw_captions = []
    # optim_state = deepcopy(model.optimizer.state_dict())
    start_time = time.time()

    with torch.no_grad():
        logging.info("Computing target image features for base model...")
        for batch in tqdm(target_loader):  # 36
            target_images, target_paths = batch
            if device is not None:
                target_images = target_images.cuda(device, non_blocking=True)
            image_features = model.encode_image(target_images)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)
        logging.info(f"All_target_image_features.shape: {len(all_image_features)},{all_image_features[0].shape}, use_time: {time.time() - start_time:.2f}s")
        
        if reward_model is not None:
            model.set_image_features(image_features=torch.cat(all_image_features, dim=0))
            logging.info(f"set_image_features for base model done, image_features.shape: {model.model.image_features.shape}")
            
            reward_model.set_image_features_with_dataloder(target_loader)
            logging.info(f"set_image_features_with_dataloder for reward model done, use_time: {time.time() - start_time:.2f}s")

        logging.info("Computing query & target features...")
        for batch in tqdm(query_loader):  # 66
            ref_images, target_images, text_with_blank, caption_only, ref_paths, answer_paths, raw_captions = batch
            if device is not None:
                ref_images = ref_images.cuda(device, non_blocking=True)
                target_images = target_images.cuda(device, non_blocking=True)
            id_split = tokenize(["*"])[0][1]
            # id_split = tokenize(["<|replace|>"])[0][1]
            for path in ref_paths:
                all_ref_paths.append(path)
            for path in answer_paths:
                all_answer_paths.append(path)
            for cap in raw_captions:
                all_raw_captions.append(cap)

            if cfg.MODEL.ADAPTATION == "source":
                query_image_features, caption_features, target_image_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features= model(x=ref_images, y=caption_only, z=target_images)
            else:
                query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features = model.forward_and_adapt(query_image=ref_images, query_text=caption_only, target_images_pool=torch.cat(all_image_features, dim=0))
                target_image_features = model.encode_image(target_images)

            # model.model.momentum_update_model()
            # model.model.reset_initial()
            # model.optimizer.load_state_dict(optim_state)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_mixture_features.append(mixture_features)
            all_lerp_features.append(lerp_features)
            all_nlerp_features.append(nlerp_features)
            all_slerp_features.append(slerp_features)
            all_slerp_features_static.append(slerp_features_static)
            all_composed_features.append(composed_features)
            all_target_image_features_for_gap.append(target_image_features)

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)


        logging.info(f"All_caption_features.shape: {len(all_caption_features)},{all_caption_features[0].shape}")
        logging.info(f"All_query_image_features.shape: {len(all_query_image_features)},{all_query_image_features[0].shape}")
        logging.info(f"All_mixture_features.shape: {len(all_mixture_features)},{all_mixture_features[0].shape}")
        logging.info(f"All_composed_features.shape: {len(all_composed_features)},{all_composed_features[0].shape}")
        logging.info(f"All_target_image_features.shape: {len(all_image_features)},{all_image_features[0].shape}")


        metric_func = partial(get_metrics_cirr,
                              image_features=torch.cat(all_image_features),  # 2297
                              reference_names=all_ref_paths,  # 4181
                              index_names=all_target_paths,  # 2297
                              target_names=all_answer_paths  # 4181
                              )
        
        feats = {'image-only': torch.cat(all_query_image_features),
                'text-only': torch.cat(all_caption_features),
                'mixture': torch.cat(all_mixture_features),
                'lerp': torch.cat(all_lerp_features),
                'nlerp': torch.cat(all_nlerp_features),
                'slerp': torch.cat(all_slerp_features),
                'slerp_static': torch.cat(all_slerp_features_static),
                'composed': torch.cat(all_composed_features)
                }

        for key, value in feats.items():
            metrics, add_output = metric_func(ref_features=value, exp_type=key)
            logging.info(f"Eval {key} Feature " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

        logging.info(f"total_use_time: {time.time() - start_time:.2f}s")
        
    return metrics, feats, torch.cat(all_target_image_features_for_gap)


def evaluate_cirr_test(model=None, reward_model=None, img2text=None, cfg=None, query_loader=None, target_loader=None, device=None):
    model.eval()
    if img2text is not None:
        img2text.eval()
    if reward_model is not None:
        reward_model.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_composed_plus_image_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_ids = [] 

    with torch.no_grad():
        logging.info("Computing target image features for base model...")
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if device is not None:
                target_images = target_images.cuda(device, non_blocking=True)
            image_features = model.encode_image(target_images)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)
        logging.info(f"All_target_image_features.shape: {len(all_image_features)},{all_image_features[0].shape}")

        if reward_model is not None:
            model.set_image_features(image_features=torch.cat(all_image_features, dim=0))
            logging.info(f"set_image_features for base model done, image_features.shape: {model.model.image_features.shape}")
            
            reward_model.set_image_features_with_dataloder(target_loader)
            logging.info(f"set_image_features_with_dataloder for reward model done, image_features.shape: {model.reward_model.image_features.shape}")

        logging.info("Computing query & target features...")
        for batch in tqdm(query_loader):
            # ref_images, text_with_blank, caption_only, ref_paths, pairids, text_with_blank_raw = batch
            ref_images, target_images, ref_text_tokens, caption_only, ref_paths, pairids, text_with_blank_raw = batch
            if device is not None:
                ref_images = ref_images.cuda(device, non_blocking=True)
                target_images = target_images.cuda(device, non_blocking=True)

            id_split = tokenize(["*"])[0][1]                        
            for ids in pairids:
                all_ids.append(ids)
            for path in ref_paths:
                all_ref_paths.append(path)

            if cfg.MODEL.ADAPTATION == "source":
                query_image_features, caption_features, target_image_features, mixture_features, composed_features = model(x=ref_images, y=caption_only, z=target_images)
            elif cfg.MODEL.ADAPTATION == "reward" or cfg.MODEL.ADAPTATION == "kd":
                query_image_features, caption_features, mixture_features, composed_features = model.forward_and_adapt(query_image=ref_images, query_text=caption_only, target_images_pool=torch.cat(all_image_features, dim=0))
            else:
                query_image_features, caption_features, target_image_features, mixture_features, composed_features, _ = model(x=ref_images, y=caption_only, z=target_images)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_features)            
            all_mixture_features.append(mixture_features)            

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        res_all = {}
        metrics_func = partial(get_cirr_testoutput, 
                               image_features=torch.cat(all_image_features),
                               reference_names=all_ref_paths,
                               index_names=all_target_paths,
                               id_names=all_ids)
        feats = {'image-only': torch.cat(all_query_image_features),
                 'text-only': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features),
                 'composed': torch.cat(all_composed_features)}        
        for key, value in feats.items():
            res_all[key] = metrics_func(ref_features=value)

    return res_all




def evaluate_coco(model=None, reward_model=None, img2text=None, cfg=None, loader=None, device=None):
    model.eval()
    if img2text is not None:
        img2text.eval()
    if reward_model is not None:
        reward_model.eval()
    if cfg.MODEL.ADAPTATION == "reward" or cfg.MODEL.ADAPTATION == "kd":
        logit_scale = model.model.clip_model.logit_scale.exp()
    else:
        logit_scale = model.model.logit_scale.exp()
    logit_scale = logit_scale.mean()
    all_target_image_features = []  
    all_query_image_features = []  
    all_mixture_features = []  
    all_lerp_features = []
    all_slerp_features = []
    all_nlerp_features = []
    all_slerp_features_static = []
    all_composed_features_with_class = []  
    all_text_full_features = [] 

    with torch.no_grad():
        # Extract target image features
        logging.info("Computing target image features for base model...")
        for batch in tqdm(loader):
            images, _, _, _, _, _, _, _= batch   
            if device is not None:
                images = images.cuda(device, non_blocking=True)
            image_features = model.encode_image(images)  # (bs, 768)
            all_target_image_features.append(image_features)
        logging.info(f"All_target_image_features.shape: {len(all_target_image_features)},{all_target_image_features[0].shape}")

        if reward_model is not None:
            model.set_image_features(image_features=torch.cat(all_target_image_features, dim=0))
            logging.info(f"set_image_features for base model done, image_features.shape: {model.model.image_features.shape}")
            
            reward_model.set_image_features_with_dataloder_coco(loader)
            logging.info(f"set_image_features_with_dataloder for reward model done, image_features.shape: {model.reward_model.image_features.shape}")


        for batch in tqdm(loader):
            images, region_images, text_full, text_with_blank, text_with_blank_query, filename, raw_text, _ = batch            
            if device is not None:
                images = images.cuda(device, non_blocking=True)
                region_images = region_images.cuda(device, non_blocking=True)

            ## Target image features 
            # target_image_features = model.encode_image(images)             
            id_split = tokenize(["*"])[0][1]
            
            if cfg.MODEL.ADAPTATION == "source":
                query_image_features, caption_features, target_image_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static = model(x=region_images, y=text_full, z=images)
                composed_features=lerp_features
            elif cfg.MODEL.ADAPTATION == "reward" or cfg.MODEL.ADAPTATION == "kd":
                query_image_features, caption_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features = model.forward_and_adapt(query_image=region_images, query_text=text_full, target_images_pool=torch.cat(all_target_image_features, dim=0))
            else:
                query_image_features, caption_features, target_image_features, mixture_features, lerp_features, nlerp_features, slerp_features, slerp_features_static, composed_features, _ = model(x=region_images, y=text_full, z=images)  


            all_text_full_features.append(caption_features)       
            all_query_image_features.append(query_image_features)
            all_mixture_features.append(mixture_features)
            all_lerp_features.append(lerp_features)
            all_nlerp_features.append(nlerp_features)
            all_slerp_features.append(slerp_features)
            all_slerp_features_static.append(slerp_features_static)
            all_composed_features_with_class.append(composed_features)  


        metric_func = partial(get_metrics_coco, 
                image_features=torch.cat(all_target_image_features), 
                logit_scale=logit_scale
                )
     
        feats = {'image-only': torch.cat(all_query_image_features),
                'text-only': torch.cat(all_text_full_features),
                'mixture': torch.cat(all_mixture_features),
                'lerp': torch.cat(all_lerp_features),
                'nlerp': torch.cat(all_nlerp_features),
                'slerp': torch.cat(all_slerp_features),
                'slerp_static': torch.cat(all_slerp_features_static),
                'composed': torch.cat(all_composed_features_with_class)
                }

        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(f"Eval {key} Feature " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics, 



def get_metrics_fashion(image_features, ref_features, target_names, answer_names, exp_type):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())

    # Compute the metrics
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics


def get_metrics_cirr(image_features, ref_features, reference_names, index_names, target_names, exp_type):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    # print(sorted_indices.shape) # torch.Size([4181, 2297])
    sorted_index_names = np.array(index_names)[sorted_indices]
    # print(image_features.shape, ref_features.shape) # torch.Size([2297, 768]) torch.Size([4181, 768])
    # print(sorted_index_names.shape, reference_names.shape, index_names.shape, target_names.shape) # (4181, 2297) (4181,) (2297,) (4181,)
    # Delete the reference image from the results
    # print(reference_names, sorted_index_names, target_names)
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names),
        len(index_names) - 1).reshape(len(target_names), -1))

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    
    add_output = {}
    add_output["reference_names"] = reference_names
    add_output["sorted_index_names"] = sorted_index_names
    add_output["target_names"] = target_names

    return metrics, add_output

def get_cirr_testoutput(image_features, ref_features, reference_names, index_names, id_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    result_dict = {"version": "rc2", "metric": "recall"}
    for ind in range(len(id_names)):
        pairid = str(id_names[ind].item())
        result_dict[pairid] = []
        for t in range(50):
            result_dict[pairid].append(sorted_index_names[ind][t].replace(".png", ""))
    return result_dict



def get_metrics_coco(image_features, ref_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale.cpu() * image_features @ ref_features.t()).detach().cpu()
    logits_per_ref = logits_per_image.t().detach().cpu()
    # logits = {"image_to_ref": logits_per_image, "ref_to_image": logits_per_ref}
    logits = {"ref_to_image": logits_per_ref}
    ground_truth = torch.arange(len(ref_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        # metrics[f"{name}_mean_rank"] = preds.mean() + 1
        # metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10, 50, 100]:
            metrics[f"R@{k}"] = np.mean(preds < k)*100
            # metrics[f"{name}_R@{k}"] = np.mean(preds < k)*100
    return metrics

