import argparse
import time

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD

from src.data_pipeline import create_dataloaders
from src.model import BasicCNN, create_resnet18
from src.utils import set_seed, get_environment_info


def get_model(model_name, num_classes=11):
    if model_name == "basic_cnn":
        return BasicCNN(num_classes=num_classes)

    if model_name == "resnet18":
        return create_resnet18(num_classes=num_classes)

    raise ValueError(f"Nepoznat model: {model_name}")


def get_optimizer(name, model, learning_rate):
    if name == "adam":
        return Adam(model.parameters(), lr=learning_rate)

    if name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate)

    if name == "sgd":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    raise ValueError(f"Nepoznat optimizer: {name}")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data"
    )

    parser.add_argument(
        "--model",
        choices=["basic_cnn", "resnet18"],
        default="basic_cnn"
    )

    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default="adam"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--augmentation",
        action="store_true"
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, validation_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_augmentation=args.augmentation
    )

    model = get_model(args.model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        args.optimizer,
        model,
        args.learning_rate
    )

    environment_info = get_environment_info()

    mlflow.set_experiment(
        "Food11 Classification"
    )

    with mlflow.start_run(
        run_name=args.run_name
    ):
        mlflow.log_param(
            "model",
            args.model
        )

        mlflow.log_param(
            "optimizer",
            args.optimizer
        )

        mlflow.log_param(
            "learning_rate",
            args.learning_rate
        )

        mlflow.log_param(
            "batch_size",
            args.batch_size
        )

        mlflow.log_param(
            "epochs",
            args.epochs
        )

        mlflow.log_param(
            "augmentation",
            args.augmentation
        )

        mlflow.log_param(
            "seed",
            args.seed
        )

        for key, value in environment_info.items():
            mlflow.log_param(
                key,
                value
            )

        start_time = time.time()

        best_validation_accuracy = 0.0

        for epoch in range(args.epochs):
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            validation_loss, validation_accuracy = evaluate(
                model,
                validation_loader,
                criterion,
                device
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch
            )

            mlflow.log_metric(
                "train_accuracy",
                train_accuracy,
                step=epoch
            )

            mlflow.log_metric(
                "validation_loss",
                validation_loss,
                step=epoch
            )

            mlflow.log_metric(
                "validation_accuracy",
                validation_accuracy,
                step=epoch
            )

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy:.4f}"
            )

        training_time = time.time() - start_time

        mlflow.log_metric(
            "best_validation_accuracy",
            best_validation_accuracy
        )

        mlflow.log_metric(
            "training_time_seconds",
            training_time
        )

        print(
            f"\nBest validation accuracy: "
            f"{best_validation_accuracy:.4f}"
        )

        print(
            f"Training time: "
            f"{training_time:.2f} seconds"
        )


if __name__ == "__main__":
    main()