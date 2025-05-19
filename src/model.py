import torchvision
import torch


def create_model():
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 13)
    model.load_state_dict(torch.load("resnet50.pth"))
    ################### OPTIONAL #########################
    model.requires_grad_(False)
    model.fc.requires_grad_(True)
    return model