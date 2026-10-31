
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("data/processed/Training_data")
MODEL_DIR = Path("models")

FEATURE_COLUMNS = [
    "jaccard_similarity",
    "user_coverage",
    "recipe_coverage",
    "matched_ingredients",
    "missing_ingredients",
    "user_ingredient_count",
    "recipe_ingredient_count",
]

TARGET_COLUMN = "label"


def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def validate_dataset(df, name):
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {missing_columns}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            f"{name} contains missing feature or label values."
        )

    labels = set(df[TARGET_COLUMN].unique())

    if labels != {0, 1}:
        raise ValueError(
            f"{name} must contain both labels 0 and 1. "
            f"Found: {labels}"
        )


def evaluate_model(model, df, dataset_name):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }

    print(f"\n{dataset_name.upper()} Results")

    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Negative", "Positive"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        )
    )

    return metrics


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset("training_examples.csv")
    validation_df = load_dataset("validation_examples.csv")
    test_df = load_dataset("test_examples.csv")

    validate_dataset(train_df, "Training dataset")
    validate_dataset(validation_df, "Validation dataset")
    validate_dataset(test_df, "Test dataset")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training Logistic Regression model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    validation_metrics = evaluate_model(
        model,
        validation_df,
        "Validation",
    )

    test_metrics = evaluate_model(
        model,
        test_df,
        "Test",
    )

    model_path = MODEL_DIR / "recipe_match_model.joblib"

    joblib.dump(model, model_path)

    metrics_path = MODEL_DIR / "evaluation_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "features": FEATURE_COLUMNS,
                "validation": validation_metrics,
                "test": test_metrics,
                "training_labels_are_synthetic": True,
            },
            file,
            indent=4,
        )

    print("\nSaved files")
    print(f"Trained model: {model_path}")
    print(f"Evaluation metrics: {metrics_path}")


if __name__ == "__main__":
    main()