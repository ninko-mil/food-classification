import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.experiments import EXPERIMENT_CONFIGS
from src.model import create_resnet18
from src.utils import set_seed


# ============================================================
# Osnovna podešavanja
# ============================================================

SEED = 42
NUM_CLASSES = 11
NUM_FOLDS = 5
EPOCHS = 5

IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Transformacije
# ============================================================

def get_transform(use_augmentation=False):
    steps = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    ]

    if use_augmentation:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10)
        ])

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    return transforms.Compose(steps)


# ============================================================
# Dataset za cross-validation
# ============================================================

class Food11FoldDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# Učitavanje training + validation podataka
# ============================================================

def collect_samples(data_dir):
    data_dir = Path(data_dir)

    training_dir = data_dir / "training"
    validation_dir = data_dir / "validation"

    classes = sorted([
        folder.name
        for folder in training_dir.iterdir()
        if folder.is_dir()
    ])

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    samples = []

    for split_dir in [training_dir, validation_dir]:
        for class_name in classes:
            class_dir = split_dir / class_name

            for image_path in class_dir.iterdir():
                if image_path.is_file():
                    samples.append(
                        (
                            image_path,
                            class_to_idx[class_name]
                        )
                    )

    return samples, classes


# ============================================================
# Optimizer
# ============================================================

def create_optimizer(model, config):
    optimizer_name = config["optimizer"].lower()
    learning_rate = config["learning_rate"]

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.get("momentum", 0.9)
        )

    raise ValueError(
        f"Nepoznat optimizer: {optimizer_name}"
    )


# ============================================================
# Jedna epoha treninga
# ============================================================

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

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Validacija
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            total += labels.size(0)

    validation_loss = running_loss / total

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return (
        validation_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# Jedna konfiguracija
# ============================================================

def run_configuration(
    config,
    samples,
    labels,
    device
):
    print()
    print("=" * 70)
    print("Konfiguracija:", config["name"])
    print("=" * 70)

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []

    configuration_start = time.time()

    with mlflow.start_run(
        run_name=config["name"]
    ):

        # ----------------------------------------------------
        # Hiperparametri
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            config["model"]
        )

        mlflow.log_param(
            "learning_rate",
            config["learning_rate"]
        )

        mlflow.log_param(
            "optimizer",
            config["optimizer"]
        )

        mlflow.log_param(
            "augmentation",
            config["augmentation"]
        )

        mlflow.log_param(
            "batch_size",
            config["batch_size"]
        )

        mlflow.log_param(
            "epochs_per_fold",
            EPOCHS
        )

        mlflow.log_param(
            "num_folds",
            NUM_FOLDS
        )

        mlflow.log_param(
            "seed",
            SEED
        )

        mlflow.log_param(
            "device",
            str(device)
        )

        if torch.cuda.is_available():
            mlflow.log_param(
                "gpu_name",
                torch.cuda.get_device_name(0)
            )

            gpu_memory_gb = (
                torch.cuda.get_device_properties(0)
                .total_memory
                / (1024 ** 3)
            )

            mlflow.log_param(
                "gpu_memory_gb",
                round(gpu_memory_gb, 2)
            )

            mlflow.log_param(
                "cuda_version",
                torch.version.cuda
            )

        # ----------------------------------------------------
        # Foldovi
        # ----------------------------------------------------

        for fold, (train_indices, val_indices) in enumerate(
            skf.split(samples, labels),
            start=1
        ):
            print()
            print(f"Fold {fold}/{NUM_FOLDS}")

            set_seed(SEED + fold)

            train_samples = [
                samples[index]
                for index in train_indices
            ]

            val_samples = [
                samples[index]
                for index in val_indices
            ]

            train_dataset = Food11FoldDataset(
                train_samples,
                transform=get_transform(
                    config["augmentation"]
                )
            )

            val_dataset = Food11FoldDataset(
                val_samples,
                transform=get_transform(False)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )

            model = create_resnet18(
                num_classes=NUM_CLASSES
            ).to(device)

            criterion = nn.CrossEntropyLoss()

            optimizer = create_optimizer(
                model,
                config
            )

            fold_start = time.time()

            last_metrics = None

            for epoch in range(1, EPOCHS + 1):
                train_loss, train_accuracy = (
                    train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device
                    )
                )

                (
                    val_loss,
                    val_accuracy,
                    val_precision,
                    val_recall,
                    val_f1
                ) = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device
                )

                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.4f}"
                )

                # --------------------------------------------
                # Loss i metrike PO EPOHI za svaki fold
                # --------------------------------------------

                mlflow.log_metric(
                    f"fold_{fold}_train_loss",
                    train_loss,
                    step=epoch
                )

                mlflow.log_metric(
                    f"fold_{fold}_train_accuracy",
                    train_accuracy,
                    step=epoch
                )

                mlflow.log_metric(
                    f"fold_{fold}_val_loss",
                    val_loss,
                    step=epoch
                )

                mlflow.log_metric(
                    f"fold_{fold}_val_accuracy",
                    val_accuracy,
                    step=epoch
                )

                last_metrics = (
                    val_accuracy,
                    val_precision,
                    val_recall,
                    val_f1
                )

            fold_time = time.time() - fold_start

            (
                fold_accuracy,
                fold_precision,
                fold_recall,
                fold_f1
            ) = last_metrics

            fold_accuracies.append(
                fold_accuracy
            )

            fold_precisions.append(
                fold_precision
            )

            fold_recalls.append(
                fold_recall
            )

            fold_f1_scores.append(
                fold_f1
            )

            # --------------------------------------------
            # Konačne metrike PO FOLDU
            # --------------------------------------------

            mlflow.log_metric(
                f"fold_{fold}_accuracy",
                fold_accuracy
            )

            mlflow.log_metric(
                f"fold_{fold}_precision",
                fold_precision
            )

            mlflow.log_metric(
                f"fold_{fold}_recall",
                fold_recall
            )

            mlflow.log_metric(
                f"fold_{fold}_f1",
                fold_f1
            )

            mlflow.log_metric(
                f"fold_{fold}_training_time_sec",
                fold_time
            )

            print(
                f"Fold {fold} završen | "
                f"Accuracy: {fold_accuracy:.4f} | "
                f"F1: {fold_f1:.4f} | "
                f"Time: {fold_time:.2f}s"
            )

            del model
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Prosečne CV metrike
        # ----------------------------------------------------

        total_time = (
            time.time() - configuration_start
        )

        mean_accuracy = np.mean(
            fold_accuracies
        )

        std_accuracy = np.std(
            fold_accuracies
        )

        mean_precision = np.mean(
            fold_precisions
        )

        mean_recall = np.mean(
            fold_recalls
        )

        mean_f1 = np.mean(
            fold_f1_scores
        )

        mlflow.log_metric(
            "cv_mean_accuracy",
            mean_accuracy
        )

        mlflow.log_metric(
            "cv_std_accuracy",
            std_accuracy
        )

        mlflow.log_metric(
            "cv_mean_precision",
            mean_precision
        )

        mlflow.log_metric(
            "cv_mean_recall",
            mean_recall
        )

        mlflow.log_metric(
            "cv_mean_f1",
            mean_f1
        )

        mlflow.log_metric(
            "total_training_time_sec",
            total_time
        )

        print()
        print(
            f"CV Mean Accuracy: "
            f"{mean_accuracy:.4f}"
        )

        print(
            f"CV Std Accuracy: "
            f"{std_accuracy:.4f}"
        )

        print(
            f"CV Mean F1: "
            f"{mean_f1:.4f}"
        )

        print(
            f"Ukupno vreme: "
            f"{total_time:.2f}s"
        )


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    mlflow.set_experiment(
        "Food11 Cross Validation"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    samples, classes = collect_samples(
        "data"
    )

    labels = np.array([
        label
        for _, label in samples
    ])

    print(
        "Broj uzoraka za cross-validation:",
        len(samples)
    )

    print(
        "Broj klasa:",
        len(classes)
    )

    print(
        "Konfiguracija za testiranje:",
        len(EXPERIMENT_CONFIGS)
    )

    for config in EXPERIMENT_CONFIGS:
        run_configuration(
            config,
            samples,
            labels,
            device
        )


if __name__ == "__main__":
    main()