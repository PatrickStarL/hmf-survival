import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
def apply_sampling(target_bag_size, all_features, all_coords):
    attn_mask = None
    if target_bag_size > 0:
        bag_size = all_features.size(0)
        attn_mask = torch.ones(bag_size)
        if bag_size < target_bag_size:
            sampled_features = torch.cat([all_features, torch.zeros(
                (target_bag_size - bag_size, all_features.shape[1]))], dim=0)
            attn_mask = torch.cat(
                [attn_mask, torch.zeros((target_bag_size - bag_size))])
            if len(all_coords) > 0:
                all_coords = np.concatenate(
                    [all_coords, np.zeros((target_bag_size - bag_size, 2))], axis=0)
        else:
            sampled_patch_ids = np.random.choice(
                np.arange(bag_size), target_bag_size, replace=False)
            sampled_features = all_features[sampled_patch_ids, :]
            attn_mask = attn_mask[:target_bag_size]
            if len(all_coords) > 0:
                all_coords = all_coords[sampled_patch_ids, :]
        all_features = sampled_features
    return all_features, all_coords, attn_mask




# Suppose you have a HuggingFace tokenizer called `tokenizer`
# and each sample['text'] is a list of strings.

def collate_fn(batch):
    collated = {}
    keys = batch[0].keys()

    for k in keys:
        values = [d[k] for d in batch]

        if k != 'text':
            # For typical numeric/tensor fields
            collated[k] = default_collate(values)
        else:
            # Tokenize text
            # e.g. each 'values[i]' is a list of text segments for sample i
            tokenized_per_sample = []
            for text_segments in values:
                tokenized_per_sample.append(text_segments)
            collated[k] = tokenized_per_sample

    return collated
