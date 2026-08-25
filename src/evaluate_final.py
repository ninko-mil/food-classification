import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.data_pipeline import load_datasets
from src.model import create_resnet18
from src.utils import get_environment_info


MODEL_PATH = Path("models/best_food11_resnet18.pth")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
NUM_CLASSES = 11


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    _, _, evaluation_dataset = load_datasets(
        "data",
        use_augmentation=False
    )

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    class_names = evaluation_dataset.classes

    model = create_resnet18(
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, labels in evaluation_loader:
            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            total_inference_time += end - start
            total_images += images.size(0)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

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

    inference_time_per_image_ms = (
        total_inference_time
        / total_images
        * 1000
    )

    model_size_mb = (
        MODEL_PATH.stat().st_size
        / (1024 ** 2)
    )

    print()
    print("FINALNI REZULTATI")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(
        f"Inference time/image: "
        f"{inference_time_per_image_ms:.4f} ms"
    )
    print(
        f"Model size: "
        f"{model_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # CSV sa glavnim metrikama
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "inference_time_ms_per_image":
                inference_time_per_image_ms,
            "model_size_mb": model_size_mb
        }
    ])

    metrics_df.to_csv(
        RESULTS_DIR / "final_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d"
    )

    plt.title(
        "Confusion Matrix - Final ResNet18"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # ROC - One-vs-Rest
    # --------------------------------------------------------

    binary_labels = label_binarize(
        all_labels,
        classes=np.arange(NUM_CLASSES)
    )

    plt.figure(
        figsize=(10, 8)
    )

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(
            binary_labels[:, i],
            all_probabilities[:, i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{class_names[i]} "
                f"(AUC={roc_auc:.2f})"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC krive - One-vs-Rest"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "roc_curves.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Precision-Recall
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    for i in range(NUM_CLASSES):
        precision_curve, recall_curve, _ = (
            precision_recall_curve(
                binary_labels[:, i],
                all_probabilities[:, i]
            )
        )

        ap = average_precision_score(
            binary_labels[:, i],
            all_probabilities[:, i]
        )

        plt.plot(
            recall_curve,
            precision_curve,
            label=(
                f"{class_names[i]} "
                f"(AP={ap:.2f})"
            )
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Precision-Recall krive"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pr_curves.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Environment / resources
    # --------------------------------------------------------

    environment_info = get_environment_info()

    with open(
        RESULTS_DIR / "resources.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for key, value in environment_info.items():
            file.write(
                f"{key}: {value}\n"
            )

    print()
    print("Sačuvani rezultati:")
    print("results/final_metrics.csv")
    print("results/confusion_matrix.png")
    print("results/roc_curves.png")
    print("results/pr_curves.png")
    print("results/resources.txt")


if __name__ == "__main__":
    main()
