from typing import Iterator
import torch
import torchvision
from PIL import Image
import os

from src.config import ConfigManager
from src.sampler import BalancedHierarchicalSampler, CombinedSampler


class Node:
    def __init__(self, name):
        self.name = name
        self._count = 0
        self.children = {}
        self._entities = []

    def add_to_node(self, path, entity, level=0):
        if level >= len(path):
            self._entities.append(entity)
            return
        part = path[level]
        if part not in self.children:
            self.children[part] = Node(path[:level+1])
        self.children[part].add_to_node(path, entity, level=level+1)
        self._count += 1

    @property
    def is_leaf(self):
        return len(self._entities) > 0

    @property
    def count(self):
        if self.is_leaf:
            return len(self._entities)
        else:
            return self._count

    @property
    def entities(self):
        if self.is_leaf:
            return list((entity, self.name) for entity in self._entities)
        else:
            child_entities = []
            for child in self.children.values():
                child_entities.extend(child.entities)
        return child_entities

    def level_iterator(self, level=None):
        """
        iterates a certain depth in a tree and returns the nodes
        """
        if level == 0:
            yield self
        elif level == None and self.is_leaf:
            yield self
        elif self.is_leaf and level != 0:
            raise Exception("Incorrect level is specified in tree.")
        else:
            if level is not None:
                level -= 1
            for child in self.children.values():
                for v in child.level_iterator(level):
                    yield v


    def print_node(self, level=0, max_level=None):
        leaves = 1
        print(' ' * (level * 4) + f"{self.name[-1]} ({self.count})")
        for node in self.children.values():
            if max_level is None or level < max_level:
                leaves += node.print_node(level + 1, max_level=max_level)
        return leaves

class HiererchicalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, level=None):
        self.tree = Node("Dataset") # keeps the group information of self.data in a tree (per index).
        self.level = level
        if level is None:
            self.level = 7  # Hardcoded
        self.classes = set()
        data = []
        index = 0
        for group_name in sorted(os.listdir(dataset_path)):
            if not os.path.isdir(os.path.join(dataset_path, group_name)):
                continue
            for image_name in sorted(os.listdir(os.path.join(dataset_path, group_name))):
                group = tuple(group_name.split("_")[1:])
                image_path = os.path.join(dataset_path, group_name, image_name)
                data.append({
                        "image_path": image_path,
                        "group": group,
                    }
                )
                self.tree.add_to_node(group, index)
                index += 1
                self.classes.add(group[:self.level])
        self.data = data
        self.classes = {group: index for (index, group) in enumerate(sorted(list(self.classes)))}
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomCrop(224, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4556, 0.4714, 0.3700), (0.2370, 0.2318, 0.2431))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]["image_path"])
        target = self.classes[self.data[idx]["group"][:self.level]]
        if self.transform:
            image = self.transform(image)

        return image, target

    def get_group_iterator(self, level=None) -> Iterator[Node]:
        for group in self.tree.level_iterator(level):
            yield group

def create_train_dataset_and_loader():
    # Create dataset with level-2 classification
    train_dataset = HiererchicalDataset(dataset_path='train', level=2)
    print("Dataset Length:", f"{len(train_dataset)}")
    train_dataset.tree.print_node(max_level=2)
    print(train_dataset.classes)
    
    # Create samplers for each level-2 class
    samplers = []
    for level2_class in range(len(train_dataset.classes)):
        # Get all level-7 groups that belong to this level-2 class
        level7_groups = []
        for group in train_dataset.tree.level_iterator(7):
            if train_dataset.classes[group.name[:2]] == level2_class:
                level7_groups.append(group.name)
        
        # Create sampler for this level-2 class
        sampler = BalancedHierarchicalSampler(train_dataset, level2_class, level7_groups)
        samplers.append(sampler)
    
    # Create the combined sampler
    combined_sampler = CombinedSampler(samplers)
    
    # Create dataloader with the combined sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=ConfigManager().get("batch_size"),
        sampler=combined_sampler,
        num_workers=ConfigManager().get("num_workers")
    )
    
    return train_dataset, train_loader
