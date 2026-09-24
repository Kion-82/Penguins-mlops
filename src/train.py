"""Train a baseline classifier on the Palmer Penguins dataset."""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, load_data, split_data


def build_model() -> Pipeline:
    """Full pipeline: preprocessing + classifier, trained and saved as one object."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()