from pathlib import Path

import mlflow
import pandas as pd
import matplotlib.pyplot as plt


EXPERIMENT_NAME = "Food11 Cross Validation"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow eksperiment '{EXPERIMENT_NAME}' nije pronađen."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    rows = []

    for _, run in runs.iterrows():
        run_name = run.get("tags.mlflow.runName", "unknown")

        row = {
            "run_name": run_name,
            "model": run.get("params.model"),
            "learning_rate": run.get("params.learning_rate"),
            "optimizer": run.get("params.optimizer"),
            "augmentation": run.get("params.augmentation"),
            "batch_size": run.get("params.batch_size"),
        }

        for fold in range(1, 6):
            row[f"fold_{fold}_accuracy"] = run.get(
                f"metrics.fold_{fold}_accuracy"
            )

        row["cv_mean_accuracy"] = run.get(
            "metrics.cv_mean_accuracy"
        )

        row["cv_std_accuracy"] = run.get(
            "metrics.cv_std_accuracy"
        )

        row["cv_mean_precision"] = run.get(
            "metrics.cv_mean_precision"
        )

        row["cv_mean_recall"] = run.get(
            "metrics.cv_mean_recall"
        )

        row["cv_mean_f1"] = run.get(
            "metrics.cv_mean_f1"
        )

        row["total_training_time_sec"] = run.get(
            "metrics.total_training_time_sec"
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by="cv_mean_accuracy",
        ascending=False
    )

    csv_path = RESULTS_DIR / "cross_validation_results.csv"
    df.to_csv(csv_path, index=False)

    print("\nCross-validation rezultati:")
    print(df.to_string(index=False))

    print(f"\nSačuvano: {csv_path}")

    # Grafikon prosečne tačnosti
    plt.figure(figsize=(10, 6))

    plt.bar(
        df["run_name"],
        df["cv_mean_accuracy"]
    )

    plt.title("Poređenje konfiguracija - CV mean accuracy")
    plt.xlabel("Konfiguracija")
    plt.ylabel("Prosečna CV tačnost")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = RESULTS_DIR / "cv_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Sačuvano: {plot_path}")


if __name__ == "__main__":
    main()

