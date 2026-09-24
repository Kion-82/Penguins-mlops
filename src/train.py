"""Train and compare classifiers on the Palmer Penguins dataset, tracked with MLflow."""
import mlflow
import mlflow.sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, load_data, split_data

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "penguins-classification"
RANDOM_STATE = 42
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE
    ),
}


def build_model(classifier: ClassifierMixin) -> Pipeline:
    """Full pipeline: preprocessing + classifier, trained and saved as one object."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", classifier),
    ])


def train_and_log(name, classifier, X_train, X_test, y_train, y_test) -> None:
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

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.dropna().head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    print(
        f"{name:<22} CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} | "
        f"Test acc: {test_acc:.4f} | Test F1: {test_f1:.4f}"
    )


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, random_state=RANDOM_STATE)

    for name, classifier in CANDIDATES.items():
        train_and_log(name, classifier, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()