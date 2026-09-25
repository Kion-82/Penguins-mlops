"""Train candidate models, pick the best, register it in MLflow and export it for serving."""
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from src.model import MODEL_DIR, SKOPS_TRUSTED_TYPES, build_model, export_model
from src.preprocess import load_data, split_data

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "penguins-classification"
REGISTERED_MODEL_NAME = "penguins-classifier"
CHAMPION_ALIAS = "champion"
RANDOM_STATE = 42

# Order matters: simpler models first, so they win ties.
CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE
    ),
}


def train_and_log(name, classifier, X_train, X_test, y_train, y_test) -> dict:
    """Cross-validate, fit, evaluate and log one candidate model as an MLflow run."""
    model = build_model(classifier)

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_params(classifier.get_params())

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
        mlflow.log_metric("cv_f1_macro_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_macro_std", cv_scores.std())

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1)

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.dropna().head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    print(
        f"{name:<22} CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} | "
        f"Test acc: {test_acc:.4f} | Test F1: {test_f1:.4f}"
    )
    return {
        "name": name,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "model": model,
        "model_uri": model_info.model_uri,
    }


def select_best(results: list[dict]) -> dict:
    """Highest CV F1 wins; ties go to lower std, then to the earlier (simpler) candidate."""
    return min(results, key=lambda r: (-round(r["cv_mean"], 4), round(r["cv_std"], 4)))


def register_champion(model_uri: str) -> str:
    """Register the model as a new version and point the 'champion' alias at it."""
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version
    )
    return version.version


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, random_state=RANDOM_STATE)

    results = [
        train_and_log(name, classifier, X_train, X_test, y_train, y_test)
        for name, classifier in CANDIDATES.items()
    ]

    best = select_best(results)
    version = register_champion(best["model_uri"])
    export_model(best["model"], input_example=X_train.dropna().head(3))

    print(f"\nBest model: {best['name']} -> registered as version {version} ({CHAMPION_ALIAS})")
    print(f"Exported to: {MODEL_DIR}")


if __name__ == "__main__":
    main()