import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random

class BalancedHierarchicalSampler(Sampler):
    def __init__(self, dataset, level2_class, level7_groups):
        """
        Args:
            dataset: The dataset to sample from
            level2_class: The level-2 class index for this sampler
            level7_groups: List of level-7 group indices that belong to this level-2 class
        """
        self.dataset = dataset
        self.level2_class = level2_class
        self.level7_groups = level7_groups
        
        # Create mapping from level-7 group to indices
        self.group_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            group = dataset.data[idx]["group"]
            if dataset.classes[group[:2]] == level2_class:  # Check if belongs to this level-2 class
                self.group_to_indices[group].append(idx)
        
        # Calculate the minimum number of samples per level-7 group
        self.min_samples = min(len(indices) for indices in self.group_to_indices.values())
        
    def __iter__(self):
        # For each level-7 group, sample min_samples indices
        indices = []
        for group_indices in self.group_to_indices.values():
            if len(group_indices) > self.min_samples:
                sampled = random.sample(group_indices, self.min_samples)
            else:
                sampled = group_indices
            indices.extend(sampled)
        
        # Shuffle the indices
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.min_samples * len(self.group_to_indices) 


class CombinedSampler(Sampler):
    def __init__(self, samplers):
        self.samplers = samplers
        self.lengths = [len(s) for s in samplers]
        self.total_length = sum(self.lengths)
        
    def __iter__(self):
        # Create iterators for each sampler
        iterators = [iter(s) for s in self.samplers]
        # Alternate between samplers
        while True:
            for it in iterators:
                try:
                    yield next(it)
                except StopIteration:
                    return
                    
    def __len__(self):
        return self.total_length
