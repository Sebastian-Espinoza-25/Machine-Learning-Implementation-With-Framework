# Sebastian Espinoza Farias A01750311
# ID3 Decision Tree using scikit-learn, entropy, one-hot-encoding, Confusion Matrix Heatmap, GridSearchCV

# IMPORTS
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# LOGIC
def main():
    parser = argparse.ArgumentParser(description="ID3 Decision Tree with scikit-learn")
    # Required argument to run: CSV path
    parser.add_argument("csv_path")
    # Optional arguments: hyperparameters, options
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Load data as strings (categoricals only, ID3-style)
    data = pd.read_csv(args.csv_path, dtype=str)
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()
    target_name = y.name if y.name is not None else data.columns[-1]

    # Strip whitespace from categoricals
    for c in X.columns:
        X[c] = X[c].astype(str).str.strip()
    y = y.astype(str).str.strip()

    # Split into train and test (validation is performed via cross-validation in GridSearchCV)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    # Save splits to CSV files
    base_stem = os.path.splitext(os.path.basename(args.csv_path))[0]
    train_out = f"{base_stem}_train.csv"
    test_out  = f"{base_stem}_test.csv"
    train_df = X_train.copy(); train_df[target_name] = y_train.values
    test_df  = X_test.copy();  test_df[target_name]  = y_test.values
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print("\nSaved splits:")
    print(f"  Train -> {train_out}  (rows: {len(train_df)})")
    print(f"  Test  -> {test_out}   (rows: {len(test_df)})")

    # Print the class distribution in train and test
    print(f"\nDataset: {args.csv_path}")
    print("Train class counts:\n", y_train.value_counts())
    print("Test class counts:\n", y_test.value_counts())

    # Preprocessing: One-hot-encoding for all feature columns (categorical pipeline)
    cat_features = list(X.columns)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ],
        remainder="drop"
    )

    # Base tree (ID3-like using entropy)
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )

    # Pipeline
    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", tree)
    ])

    # GridSearchCV (Always on) – performs validation via cross-validation on the training set
    n_splits = safe_stratified_cv(y_train, max_k=5)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)

    param_grid = {
        "model__max_depth": [3, 4, 5, None],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3, 5e-3],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_train, y_train)
    print(f"\nGridSearchCV used StratifiedKFold(n_splits={n_splits})")
    print("Best params from GridSearchCV:", grid.best_params_)
    print(f"Best CV score: {grid.best_score_:.3f}")
    pipe = grid.best_estimator_

    # Pretty print tree
    pretty_print_tree(pipe, cat_features)

    # Predict
    y_train_pred = pipe.predict(X_train)
    y_test_pred  = pipe.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)

    print("\nAccuracy")
    print("  Train:", train_acc)
    print("  Test :", test_acc)

    # === Bias diagnosis (NEW) ===
    bias_level, bias_text = diagnose_bias(train_acc)
    print("\nBias diagnosis:", bias_level)
    print("Explanation:", bias_text)

    # === Train vs Test accuracy bar chart (NEW) ===
    plot_train_test_accuracy(train_acc, test_acc, title="Accuracy: Train vs Test")

    # Confusion matrix (test) console output
    print("\nConfusion Matrix (Test)")
    cm_df = confusion_matrix_df(y_test, y_test_pred)
    print(cm_df)

    # Classification report
    print("\nPer-class Precision, Recall, F1 (Test)")
    print(classification_report(y_test, y_test_pred, digits=3, zero_division=0))

    # Confusion matrix heatmap (Test) plot
    classes_sorted = sorted(y.unique())
    plot_confusion_matrix(y_test, y_test_pred, classes=classes_sorted,
                          title="Confusion Matrix (Test)", out_png=None)

# PRETTY-PRINT
def pretty_print_tree(trained_pipe: Pipeline, cat_features):
    """Pretty-print the learned tree using OHE feature names."""
    ohe = trained_pipe.named_steps["preprocess"].named_transformers_["cat"]
    feat_names = ohe.get_feature_names_out(cat_features)
    tree_text = export_text(trained_pipe.named_steps["model"], feature_names=list(feat_names))
    print("\nPretty tree (train):")
    print(tree_text)

# HELPER-FUNCTIONS
def confusion_matrix_df(y_true, y_pred):
    """Return a labeled confusion matrix DataFrame for console printing."""
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm,
                        index=[f"true_{l}" for l in labels],
                        columns=[f"pred_{l}" for l in labels])

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix (Test)", out_png=None):
    """Plot a heatmap of the confusion matrix for the test set."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df_plot = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df_plot, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f"Saved confusion matrix figure -> {out_png}")
    plt.show()

def safe_stratified_cv(y, max_k=5):
    """Return a safe number of CV splits based on the smallest class count."""
    min_per_class = y.value_counts().min()
    return max(2, min(max_k, int(min_per_class)))

# === NEW HELPERS FOR BIAS AND ACCURACY PLOT ===

def diagnose_bias(train_acc):
    """
    Diagnose bias level using training accuracy only:
    - High bias:    train_acc < 0.60
    - Medium bias:  0.60 <= train_acc < 0.80
    - Low bias:     train_acc >= 0.80
    """
    if train_acc < 0.60:
        level = "HIGH"
        rationale = ("Low training accuracy (<60%). The model fails to capture the underlying patterns "
                     "(likely underfitting / high bias).")
    elif train_acc < 0.80:
        level = "MEDIUM"
        rationale = ("Moderate training accuracy (60–80%). The model learns some patterns but still shows "
                     "systematic errors (moderate bias).")
    else:
        level = "LOW"
        rationale = ("High training accuracy (≥80%). The model captures the relationship between variables "
                     "well (low bias).")
    return level, rationale

def plot_train_test_accuracy(train_acc, test_acc, title="Accuracy: Train vs Test"):
    """Bar chart comparing Train vs Test accuracy."""
    df_plot = pd.DataFrame({
        "split": ["Train", "Test"],
        "accuracy": [train_acc, test_acc]
    })
    plt.figure(figsize=(5, 4))
    sns.barplot(data=df_plot, x="split", y="accuracy")
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# To run from command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python id3_sklearn.py <dataset.csv>")
        sys.exit(1)
    main()
