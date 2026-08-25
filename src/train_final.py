import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from src.data_pipeline import load_datasets
from src.model import create_resnet18
from src.utils import set_seed, get_environment_info


SEED = 42
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.0001

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_food11_resnet18.pth"


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Učitavanje train/validation/evaluation
    train_dataset, validation_dataset, _ = load_datasets(
        "data",
        use_augmentation=True
    )

    # Spajamo training + validation
    final_train_dataset = ConcatDataset([
        train_dataset,
        validation_dataset
    ])

    train_loader = DataLoader(
        final_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    model = create_resnet18(
        num_classes=11
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    mlflow.set_experiment(
        "Food11 Final Model"
    )

    environment_info = get_environment_info()

    with mlflow.start_run(
        run_name="final_resnet18_adamw"
    ):

        mlflow.log_param("model", "resnet18")
        mlflow.log_param("optimizer", "adamw")
        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE
        )
        mlflow.log_param(
            "batch_size",
            BATCH_SIZE
        )
        mlflow.log_param(
            "epochs",
            EPOCHS
        )
        mlflow.log_param(
            "augmentation",
            True
        )
        mlflow.log_param(
            "seed",
            SEED
        )

        for key, value in environment_info.items():
            mlflow.log_param(
                key,
                value
            )

        start_time = time.time()

        best_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                running_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = outputs.argmax(
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

            epoch_loss = (
                running_loss / total
            )

            epoch_accuracy = (
                correct / total
            )

            mlflow.log_metric(
                "train_loss",
                epoch_loss,
                step=epoch
            )

            mlflow.log_metric(
                "train_accuracy",
                epoch_accuracy,
                step=epoch
            )

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: {epoch_accuracy:.4f}"
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss

                torch.save(
                    model.state_dict(),
                    MODEL_PATH
                )

        total_time = (
            time.time() - start_time
        )

        mlflow.log_metric(
            "training_time_sec",
            total_time
        )

        mlflow.log_metric(
            "best_train_loss",
            best_loss
        )

        print()
        print(
            "Finalni model sačuvan:",
            MODEL_PATH
        )

        print(
            "Ukupno vreme:",
            f"{total_time:.2f}s"
        )


if __name__ == "__main__":
    main()



