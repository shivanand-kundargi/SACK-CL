import sys
sys.path.insert(0, "/g/g91/kundargi1/SACK")
import time
from functools import partial
from typing import Callable, Tuple, List

import numpy as np
import torch
from math import ceil
from torch import Tensor
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets.cifar import CIFAR100

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from utils.concept_paths import PATHS_CONFIG

import random
import wandb
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import datetime
import json
import pandas as pd
import torch
from utils import utils_cd
from utils import similarity
from tqdm import tqdm
import math
import pandas as pd
import clip
import numpy as np
import json
# import args
import logging

from utils import create_if_not_exists, smart_joint
from utils.conf import base_path

# args=args.add_initial_args(parser=argparse.ArgumentParser())


class IcarlDissectandScore:
    def __init__(self, args, model, protocol, similarity_fn, class_order, curent_experience, current_experience_classes, device):
        self.model = model
        self.protocol = protocol
        self.similarity_fn = similarity_fn
        self.device = device
        self.experience_number = curent_experience
        self.similarity_fn = similarity_fn
        self.outputs = {"layer":[], "unit":[], "description":[], "similarity":[]} 
        self.args = args
        self.cache_root = None
        self.activation_save_dir = None
        self.task_results_dir = None
        self._last_clip_save_name = None
        self._last_target_layer = None
        self._last_concept_path = None
        self._last_words = None
        self.sample_concept_dictionary_path = None

    def _sanitize_path_component(self, value):
        value = str(value)
        safe_chars = []
        for char in value:
            if char.isalnum() or char in ('-', '_', '.'):
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        sanitized = ''.join(safe_chars).strip('._')
        return sanitized or "default"

    def _get_cache_root(self, save_dir: str):
        cache_group = self._sanitize_path_component(save_dir or "sack_cache")
        variant = self._sanitize_path_component(
            getattr(self.args, "sack_schedule_variant", f"type_{getattr(self.args, 'sack_scores_type', 'na')}")
        )
        run_name = "{}_{}_{}_seed{}".format(
            self._sanitize_path_component(self.args.model),
            self._sanitize_path_component(self.args.dataset),
            variant,
            self.args.seed,
        )
        cache_root = smart_joint(base_path(), self.args.results_path, cache_group, run_name)
        create_if_not_exists(cache_root)
        return cache_root

    def dissect(self, save_dir: str):
        cache_root = self._get_cache_root(save_dir)
        activation_save_dir = smart_joint(cache_root, "saved_activations")
        task_results_dir = smart_joint(cache_root, f"results_task{self.experience_number+1}")
        create_if_not_exists(activation_save_dir)
        create_if_not_exists(task_results_dir)
        self.cache_root = cache_root
        self.activation_save_dir = activation_save_dir
        self.task_results_dir = task_results_dir
        logging.info("SACK: task %s cache root: %s", self.experience_number + 1, cache_root)

        if self.args.dataset == "seq-cifar100" or self.args.dataset == "seq-cifar100-224" or self.args.dataset == "seq-cifar10":
            print("using seq-cifar100")
            lst=[]
            for layer_name, layer in self.model.named_children():
                lst.append(layer_name)
                # print(layer_name)
            if "last" not in lst:
                classifier=str(lst[-1])
            else:    
                classifier="last"
            path = PATHS_CONFIG["seq-cifar100"]
        
        if self.args.dataset == "seq-inaturalist-300":
            print("using seq-inaturalist-300")
            lst=[]
            for layer_name, layer in self.model.named_children():
                lst.append(layer_name)
                # print(layer_name)
            if "last" not in lst:
                classifier=str(lst[-1])
            else:    
                classifier="last"
            path = PATHS_CONFIG["seq-inaturalist-300"]

        
            # path = "/home/shivank2/gokhale_user/shivanand/mammoth/concept_sets/OG_order_cifar100_concepts"
        if self.args.dataset == "seq-cub200":
            print("using seq-cub20$seed0")
            lst=[]
            for layer_name, layer in self.model.named_children():
                lst.append(layer_name)
                # print(layer_name)
            if "last" not in lst:
                classifier=str(lst[-1])
            else:    
                classifier="last"
            path = PATHS_CONFIG["seq-cub200"]    

        if self.args.dataset == "seq-core50":
            print("using seq-core50")
            lst=[]
            for layer_name, layer in self.model.named_children():
                lst.append(layer_name)
                # print(layer_name)
            if "last" not in lst:
                classifier=str(lst[-1])
            else:    
                classifier="last"
            path = PATHS_CONFIG["seq-cub200"]    
            # path = "/home/shivank2/gokhale_user/shivanand/mammoth/concept_sets/OG_order_CUB200_concepts"    
        if self.args.dataset == "seq-imagenet-r":
            print("using seq-imagenet-r")
            lst=[]
            for layer_name, layer in self.model.named_children():
                lst.append(layer_name)
                # print(layer_name)
            if "last" not in lst:
                classifier=str(lst[-1])
            else:    
                classifier="last"  
            path = PATHS_CONFIG["seq-imagenet-r"]          
            # path = "/home/shivank2/gokhale_user/shivanand/mammoth/concept_sets/OG_order_Imagenet_R_concepts"

        # if self.args.dataset == "perm-mnist":
        #     print("using perm-mnist")
        #     # lst=[]
        #     # for layer_name, layer in self.model.named_children():
        #     #     lst.append(layer_name)
        #     #     print(layer_name)
        #     # if "last" not in lst:
        #     #     classifier=str(lst[-1])
        #     # else:    
        #     classifier="classifier"        
        #     path="/home/shivank2/gokhale_user/shivanand/mammoth/concept_sets/mnist"    

        concept_set_path = f"{path}/new_exp{self.experience_number+1}_filtered_new.txt"
        self._last_concept_path = concept_set_path

        utils_cd.save_activations(clip_name = "ViT-B/16", protocol=self.protocol, target_name = f"{self.args.model}_{self.args.dataset}",
                        target_layers = [classifier], d_probe = self.experience_number, experience_num = self.experience_number, model=self.model,
                        concept_set = concept_set_path, batch_size = 200,
                        device = "cuda", pool_mode="avg",
                        save_dir = activation_save_dir)
        # total_samples = 0
        target_layer= classifier
        outputs = {"layer":[], "unit":[], "description":[], "similarity":[]}
        with open(concept_set_path, 'r') as f:
            words = (f.read()).split('\n')
            # print(len(words))
        self._last_words = words


        save_names = utils_cd.get_save_names(clip_name = "ViT-B/16", target_name = f"{self.args.model}_{self.args.dataset}",
                                    target_layer = classifier, d_probe = self.experience_number,experience_number = self.experience_number, model= self.model,
                                    concept_set = concept_set_path, pool_mode = "avg",
                                    save_dir = activation_save_dir)
        target_save_name, clip_save_name, text_save_name = save_names
        self._last_clip_save_name = clip_save_name
        self._last_target_layer = target_layer
        similarities = utils_cd.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, self.similarity_fn, return_target_feats=False, device="cuda"
        )
        vals, ids = torch.max(similarities, dim=1)
        # print(ids)
        del similarities
        torch.cuda.empty_cache()

        descriptions = [words[int(idx)] for idx in ids]

        outputs["unit"].extend([i for i in range(len(vals))])
        outputs["layer"].extend([target_layer]*len(vals))
        outputs["description"].extend(descriptions)
        outputs["similarity"].extend(vals.cpu().numpy())

        df = pd.DataFrame(outputs)
        df.to_csv(os.path.join(task_results_dir, "descriptions.csv"), index=False)
        print("Hola! Dissected succesfully")
        self.outputs = outputs
        return outputs
    
    def get_values_from_json(self,json_file, indexes):

        with open(json_file, 'r') as file:
            data = json.load(file)

        keys = list(data.keys())

        selected_values = [data[keys[i]] for i in indexes if i < len(keys)]
        selected_values_dict = dict(zip([keys[i] for i in indexes if i < len(keys)], selected_values))

        return selected_values
    def get_clip_text_features(self,model, text, batch_size=1000):
        text_features = []
        # print("hey there, computing clip embeddings for concepts started")
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(text)/batch_size))):
                text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
        text_features = torch.cat(text_features, dim=0)
        # print("Tatastu, computing clip embeddings for concepts ended")
        return text_features


        return selected_values
    def get_clip_text_features(self,model, text, batch_size=1000):
        text_features = []
        # print("hey there, computing clip embeddings for concepts started")
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(text)/batch_size))):
                text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
        text_features = torch.cat(text_features, dim=0)
        # print("Tatastu, computing clip embeddings for concepts ended")
        return text_features

    def _filtered_concept_list(self):
        descriptions = list(self.outputs.get("description", []))
        similarity_scores = np.asarray(self.outputs.get("similarity", []), dtype=float)
        if len(descriptions) == 0 or similarity_scores.size == 0:
            return []

        percentile = getattr(self.args, 'sack_similarity_percentile', 75.0)
        try:
            percentile = float(percentile)
        except (TypeError, ValueError):
            percentile = 75.0
        percentile = max(0.0, min(100.0, percentile))
        threshold = np.percentile(similarity_scores, percentile)

        filtered_concepts = []
        seen = set()
        for desc, sim in zip(descriptions, similarity_scores):
            desc = str(desc).strip()
            if desc and sim > threshold and desc not in seen:
                filtered_concepts.append(desc)
                seen.add(desc)

        if filtered_concepts:
            return filtered_concepts

        # Degenerate but possible when all similarities tie at the percentile.
        sorted_indices = np.argsort(similarity_scores)[::-1]
        for idx in sorted_indices:
            desc = str(descriptions[int(idx)]).strip()
            if desc and desc not in seen:
                filtered_concepts.append(desc)
                seen.add(desc)
            if filtered_concepts:
                break
        return filtered_concepts

    def _concept_list(self, concepts):
        if concepts is None:
            return []
        if isinstance(concepts, str):
            values = [concepts]
        elif isinstance(concepts, dict):
            values = list(concepts.values())
        else:
            values = list(concepts)

        clean_values = []
        for value in values:
            if isinstance(value, (list, tuple)):
                clean_values.extend([str(item).strip() for item in value if str(item).strip()])
            else:
                value = str(value).strip()
                if value:
                    clean_values.append(value)
        return clean_values

    def _encode_text_concepts(self, clip_model, concepts, device, batch_size=1000):
        concepts = self._concept_list(concepts)
        if len(concepts) == 0:
            return None
        text = clip.tokenize(["{}".format(word) for word in concepts]).to(device=device)
        text_features = self.get_clip_text_features(clip_model, text, batch_size).to(device).float()
        return torch.nn.functional.normalize(text_features, p=2, dim=-1)

    def _sample_image_feature_cache_path(self):
        if self.activation_save_dir is None:
            cache_root = self._get_cache_root("ours_saved_activation")
            self.activation_save_dir = smart_joint(cache_root, "saved_activations")
            create_if_not_exists(self.activation_save_dir)
        return os.path.join(
            self.activation_save_dir,
            f"{self.experience_number}_sample_clip_image_features_vitb32_clipnorm.pt"
        )

    def _prepare_clip_images(self, batch, device):
        if isinstance(batch, (tuple, list)) and len(batch) > 2 and isinstance(batch[2], torch.Tensor):
            images = batch[2]
        else:
            images = batch[0]

        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = images.float()
        if images.shape[-2:] != (224, 224):
            images = torch.nn.functional.interpolate(
                images,
                size=(224, 224),
                mode="bicubic",
                align_corners=False
            )

        images = images.to(device, non_blocking=True)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        return (images - mean) / std

    def _load_or_encode_sample_image_features(self, clip_model, device):
        save_name = self._sample_image_feature_cache_path()
        if utils_cd._can_load_torch_file(save_name):
            logging.info("SACK: reusing sample-level CLIP image features from %s", save_name)
            return torch.load(save_name, map_location='cpu').float()

        logging.info("SACK: encoding sample-level CLIP image features for task %s", self.experience_number + 1)
        batch_size = int(max(1, getattr(self.args, "sack_sample_score_batch_size", 128)))
        pin_memory = str(device).startswith("cuda")
        data_loader = DataLoader(self.protocol, batch_size=batch_size, num_workers=0, pin_memory=pin_memory)
        all_features = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                images = self._prepare_clip_images(batch, device)
                features = clip_model.encode_image(images)
                all_features.append(features.detach().cpu())

        image_features = torch.cat(all_features, dim=0).float()
        utils_cd._atomic_torch_save(image_features, save_name)
        return image_features

    def _to_int_label(self, label):
        if isinstance(label, torch.Tensor):
            return int(label.item())
        if hasattr(label, "item"):
            return int(label.item())
        return int(label)

    def _aggregate_similarity_matrix(self, similarity_matrix):
        aggregation = str(getattr(self.args, "sack_aggregation", "max-mean")).strip().lower().replace('_', '-')
        num_class_concepts = similarity_matrix.size(-1)

        if aggregation == "max-mean":
            per_activation_scores = torch.max(similarity_matrix, dim=-1).values
            return torch.mean(per_activation_scores)
        elif aggregation == "mean-mean":
            per_activation_scores = torch.mean(similarity_matrix, dim=-1)
            return torch.mean(per_activation_scores)
        elif aggregation == "min-mean":
            per_activation_scores = torch.min(similarity_matrix, dim=-1).values
            return torch.mean(per_activation_scores)
        elif aggregation == "top3-mean":
            if num_class_concepts < 3:
                per_activation_scores = torch.mean(similarity_matrix, dim=-1)
            else:
                per_activation_scores = torch.topk(similarity_matrix, k=3, dim=-1).values.mean(dim=-1)
            return torch.mean(per_activation_scores)
        elif aggregation == "top5-mean":
            if num_class_concepts < 5:
                per_activation_scores = torch.mean(similarity_matrix, dim=-1)
            else:
                per_activation_scores = torch.topk(similarity_matrix, k=5, dim=-1).values.mean(dim=-1)
            return torch.mean(per_activation_scores)
        elif aggregation == "softmax-sharp":
            weights = torch.softmax(similarity_matrix / 0.1, dim=-1)
            per_activation_scores = torch.sum(weights * similarity_matrix, dim=-1)
            return torch.mean(per_activation_scores)
        elif aggregation == "softmax-smooth":
            weights = torch.softmax(similarity_matrix / 1.0, dim=-1)
            per_activation_scores = torch.sum(weights * similarity_matrix, dim=-1)
            return torch.mean(per_activation_scores)
        elif aggregation == "max-max":
            per_activation_scores = torch.max(similarity_matrix, dim=-1).values
            return torch.max(per_activation_scores)

        raise ValueError(f"Unsupported SACK aggregation: {aggregation}")

    def sample_scoring_function(self, clip_model, next_exp_concept_set_path, indices,
                                labels, class_labels, device, topk=5):
        filtered_concepts = self._filtered_concept_list()
        next_exp_concept_set = self.get_values_from_json(next_exp_concept_set_path, indices)
        sample_labels = [self._to_int_label(label) for label in labels]
        class_labels = [self._to_int_label(label) for label in class_labels]

        try:
            topk = int(topk)
        except (TypeError, ValueError):
            topk = 5
        topk = max(1, topk)

        if len(filtered_concepts) == 0:
            logging.warning("SACK: no filtered previous concepts found; sample-level weights fall back to uniform.")
            return [1.0 for _ in sample_labels], {}

        previous_text_features = self._encode_text_concepts(clip_model, filtered_concepts, device)
        if previous_text_features is None:
            logging.warning("SACK: no previous concept embeddings found; sample-level weights fall back to uniform.")
            return [1.0 for _ in sample_labels], {}

        class_concepts = [self._concept_list(concepts) for concepts in next_exp_concept_set]
        class_text_features = []
        for concepts in class_concepts:
            class_text_features.append(self._encode_text_concepts(clip_model, concepts, device))

        label_to_pos = {label: pos for pos, label in enumerate(class_labels)}
        image_features = self._load_or_encode_sample_image_features(clip_model, device)
        if image_features.size(0) != len(sample_labels):
            raise RuntimeError(
                f"SACK sample feature count ({image_features.size(0)}) does not match dataset size ({len(sample_labels)})."
            )
        image_features = torch.nn.functional.normalize(image_features.to(device), p=2, dim=-1)

        scores_list = []
        sample_dictionary = {}
        for sample_idx, label in enumerate(sample_labels):
            class_pos = label_to_pos.get(label)
            if class_pos is None or class_pos >= len(class_concepts):
                scores_list.append(1.0)
                sample_dictionary[str(sample_idx)] = {
                    "label": label,
                    "class_index": None,
                    "selected_concepts": [],
                    "image_concept_scores": [],
                    "sack_score": 1.0,
                }
                continue

            concepts = class_concepts[class_pos]
            text_features = class_text_features[class_pos]
            if text_features is None or len(concepts) == 0:
                scores_list.append(1.0)
                sample_dictionary[str(sample_idx)] = {
                    "label": label,
                    "class_index": int(indices[class_pos]) if class_pos < len(indices) else None,
                    "selected_concepts": [],
                    "image_concept_scores": [],
                    "sack_score": 1.0,
                }
                continue

            image_to_concept = torch.mm(image_features[sample_idx:sample_idx + 1], text_features.T).squeeze(0)
            current_topk = min(topk, image_to_concept.numel())
            top_values, top_indices = torch.topk(image_to_concept, k=current_topk)
            selected_text_features = text_features[top_indices]
            similarity_matrix = torch.mm(previous_text_features, selected_text_features.T)
            score = self._aggregate_similarity_matrix(similarity_matrix)
            score_value = float(score.item()) if isinstance(score, torch.Tensor) else float(score)
            scores_list.append(score_value)

            top_indices_cpu = top_indices.detach().cpu().tolist()
            top_values_cpu = top_values.detach().cpu().float().tolist()
            sample_dictionary[str(sample_idx)] = {
                "label": label,
                "class_index": int(indices[class_pos]) if class_pos < len(indices) else None,
                "selected_concepts": [concepts[int(idx)] for idx in top_indices_cpu],
                "image_concept_scores": [float(value) for value in top_values_cpu],
                "sack_score": score_value,
            }

        if int(getattr(self.args, "sack_sample_dump_dictionary", 1)) == 1:
            aggregation = self._sanitize_path_component(getattr(self.args, "sack_aggregation", "max-mean"))
            out_dir = self.task_results_dir or self.cache_root or self._get_cache_root("ours_saved_activation")
            create_if_not_exists(out_dir)
            output_path = os.path.join(out_dir, f"sample_concept_dictionary_top{topk}_{aggregation}.json")
            payload = {
                "metadata": {
                    "dataset": getattr(self.args, "dataset", None),
                    "model": getattr(self.args, "model", None),
                    "task": int(self.experience_number + 1),
                    "topk": int(topk),
                    "aggregation": getattr(self.args, "sack_aggregation", "max-mean"),
                    "filtered_previous_concept_count": int(len(filtered_concepts)),
                    "num_samples": int(len(sample_labels)),
                    "num_classes": int(len(class_labels)),
                },
                "samples": sample_dictionary,
            }
            with open(output_path, "w") as file:
                json.dump(payload, file, indent=2)
            self.sample_concept_dictionary_path = output_path
            logging.info("SACK: wrote sample-level concept dictionary to %s", output_path)

        return scores_list, sample_dictionary

    def scoring_function(self,clip_model,filtered_concept_set, next_exp_concept_set_path, indices, device):
        filtered_concept_set = self._filtered_concept_list()
        next_exp_concept_set = self.get_values_from_json(next_exp_concept_set_path, indices)
        if len(filtered_concept_set) == 0:
            logging.warning("SACK: no filtered previous concepts found; class-level weights fall back to uniform.")
            return [torch.tensor(1.0, device=device) for _ in next_exp_concept_set]

        text_features = self._encode_text_concepts(clip_model, filtered_concept_set, device)
        # print(text_features.shape)
        scores_list = []
        for i in next_exp_concept_set:
            next_exp_text_features = self._encode_text_concepts(clip_model, i, device)
            if next_exp_text_features is None:
                scores_list.append(torch.tensor(1.0, device=device))
                continue
            similarity_matrix = torch.mm(text_features, next_exp_text_features.T)

            score = self._aggregate_similarity_matrix(similarity_matrix)
            scores_list.append(score)
        # print(scores_list)    
        return scores_list        

        



# if __name__ == "__main__":
        # text = clip.tokenize(["{}".format(word) for word in filtered_concept_set]).to(device= device)
        # text_features = self.get_clip_text_features(clip_model, text, 1000)
        # # print(text_features.shape)
        # scores_list = []
        # for i in next_exp_concept_set:
        #     next_exp_txt  = clip.tokenize(["{}".format(word) for word in i]).to(device= device)   
        #     next_exp_text_features = self.get_clip_text_features(clip_model, next_exp_txt, 1000)
        #     # print(next_exp_text_features.shape)

        #     text_features = text_features.to(device)
        #     next_exp_text_features = next_exp_text_features.to(device)
        #     normalized_text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        #     normalized_next_exp_text_features = torch.nn.functional.normalize(next_exp_text_features, p=2, dim=-1)
        #     similarity_matrix = torch.mm(normalized_text_features, normalized_next_exp_text_features.T)

        #     max_vector = torch.max(similarity_matrix, dim=-1).values
        #     # mean_vector = torch.mean(similarity_matrix, dim=-1)
        #     # score = torch.mean(mean_vector)
        #     score = torch.mean(max_vector)
        #     # score = torch.max(max_vector)
        #     scores_list.append(score)
        # return scores_list        

        



# if __name__ == "__main__":

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761)),  
    
#     ])

#     tmp =               [87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
#                             94, 92, 10, 72, 49, 78, 61, 14,  8, 86,
#                             84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
#                             69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
#                             17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
#                             1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
#                             38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
#                             60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
#                             40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
#                             98, 13, 99,  7, 34, 55, 54, 26, 35, 39]
        

#     fixed_class_order = tmp
#     per_pixel_mean = get_dataset_per_pixel_mean(CIFAR100('./data/cifar100', train=True, download=True,
#                                                             transform=transform))
#     transform_prototypes = transforms.Compose([
#         icarl_cifar100_augment_data,
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         lambda img_pattern: img_pattern - per_pixel_mean,  
#     ])
#     protocol = NCProtocol(CIFAR100('./data/cifar100', train=True, download=True, transform=transform),
#                             CIFAR100('./data/cifar100', train=False, download=True, transform=transform_test),
#                             n_tasks=100//10, shuffle=True, seed=None, fixed_class_order=fixed_class_order)

#     task_info: NCProtocolIterator
#     train_dataset: Dataset
#     similarity_fn = similarity.soft_wpmi
#     current_experience= 9
#     ICaRL_Dissect= IcarlDissectandScore(model=None,protocol=protocol, similarity_fn=similarity_fn, class_order=fixed_class_order, curent_experience=current_experience, current_experience_classes= fixed_class_order[current_experience*10 : current_experience*10 - (current_experience*10 -10) ], device=device)
#     next_exp_concept_set_path = "/home/shivank2/gokhale_user/shivanand/icarl-pytorch/data/concept_sets/decider_cifar100_concepts.json"
#     indices = [94, 92, 10, 72, 49, 78, 61, 14,  8, 86]
#     print(ICaRL_Dissect.dissect(save_dir="saved_activations"))
#     scores = ICaRL_Dissect.scoring_function(clip_model,filtered_concept_set=None, next_exp_concept_set_path=next_exp_concept_set_path, indices=indices, device=device)
#     scores = [i.item() for i in scores]
#     print(scores)
