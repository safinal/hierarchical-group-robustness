# ⭐️ Hierarchical Group Robustness
*Contributors: [Ali Nafisi](https://safinal.github.io/)*

[![pytorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🔍 Overview
This repository contains my solution for the Hierarchical Group Robustness Challenge, part of the [**Rayan International AI Contest**](https://ai.rayan.global). The challenge focuses on training robust models that can handle hierarchical group structures in the data, specifically using a modified version of the [iNat2021](https://github.com/visipedia/inat_comp/tree/master/2021) dataset.

## 🎯 Challenge Objective
The goal is to build a robust image classification system that can:

- **Handle hierarchical group structures** in the training data, where each data point has multiple labels corresponding to its predecessors in the hierarchy
- **Train a level-2 classifier** that achieves the best accuracy across all species groups
- **Process input consisting of:**
  - Training images from the modified-iNat dataset
  - Hierarchical group information for each image
  - Image transformations for data augmentation

## ⚙️ Constraints

### Model Architecture
- **Required Backbone**: ResNet50 must be used as the feature extractor
- **Allowed Modifications**: Only changes to the linear classification head are permitted
- **Training Flexibility**: While the training time is balanced around training only the classification head, you are allowed to train the feature extractor as well
- **Submission Format**: Only model weights can be submitted, not the architecture


## 🧠 The Approach
The approach leverages hierarchical group sampling and robust training techniques to handle group shifts in the data. Here's an overview of the method:

### **Hierarchical Group Sampling**
- Implements custom samplers to handle hierarchical group structures
- Balances sampling across different group levels
- Ensures fair representation of all groups during training

## 📊 Evaluation
The model's performance is evaluated using a balanced test set across 10,000 species (level-7 groups). The evaluation metric focuses on:

- Classification accuracy at level-2 across all level-7 groups
- Performance on the K groups with the lowest accuracies, where K is a secret value between 10% and 50% of the total number of groups
- The final metric is the average accuracy of these K worst-performing groups

## 🏃🏻‍♂️‍➡️ Steps to Set Up and Run

Follow these instructions to set up your environment and execute the training pipeline.

### 1. Clone the Repository
```bash
git clone https://github.com/safinal/hierarchical-group-robustness.git
cd hierarchical-group-robustness
```

### 2. Set Up the Environment
We recommend using a virtual environment to manage dependencies.

Using ```venv```:
```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
```

Using ```conda```:
```bash
conda create --name hierarchical-group-robustness python=3.8 -y
conda activate hierarchical-group-robustness
```

### 3. Install Dependencies
Install all required libraries from the ```requirements.txt``` file:
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
First, download the modified iNat dataset from Hugging Face:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='RayanAi/inat_train_modified', filename="inat_train_modified.tar.gz", repo_type="dataset", local_dir=".")
```

Then:
- Extract the downloaded tar.gz file
- Update the configuration file with your data paths

### 5. Prepare Pretrained Checkpoint
Download the pretrained ResNet50 checkpoint:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='RayanAi/resnet50-pretrained-inat', filename="resnet50.pth", repo_type="model", local_dir="./checkpoints")
```

### 6. Run Training
```bash
python run.py --config ./config/cfg.yaml
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.