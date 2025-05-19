import torch
import argparse

from src.model import create_model
from src.config import ConfigManager
from src.train import train_model
from src.dataset import create_train_dataset_and_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Rayan International AI Contest: Backdoored Model Detection")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to the config file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_path = args.config
    config = ConfigManager(config_path)  # Initialize the singleton with the config file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, train_loader = create_train_dataset_and_loader()
    model = create_model()
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=ConfigManager().get("lr"))
    train_model(model, train_loader, criterion, optimizer, device, ConfigManager().get("num_epochs"))


if __name__ == "__main__":
    main()
