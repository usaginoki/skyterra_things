import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import os
import argparse
import joblib
import time


def load_data(features_path, labels_path):
    """
    Load features and labels from .npy files
    """
    # Load features and labels from .npy files
    X = np.load(features_path)
    y = np.load(labels_path)

    return X, y


def train_mlp_with_grid_search(X, y, output_dir=None):
    """
    Train an MLP classifier with grid search for hyperparameter tuning
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid for grid search
    param_grid = {
        "hidden_layer_sizes": [(100,), (200,), (100, 50), (200, 100), (200, 100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],  # L2 regularization
        "learning_rate_init": [0.001, 0.01],
        "dropout": [
            0.1,
            0.2,
            0.3,
        ],  # We'll handle this manually since MLPClassifier doesn't have dropout
    }

    # Using a simpler grid first to estimate total time
    simple_grid = {
        "hidden_layer_sizes": [(100,), (100, 50)],
        "activation": ["relu"],
        "alpha": [0.001],
        "learning_rate_init": [0.001],
    }

    print("Running initial grid search to estimate time...")
    mlp = MLPClassifier(max_iter=100, random_state=42, early_stopping=True)
    grid_search = GridSearchCV(mlp, simple_grid, cv=3, n_jobs=-1, verbose=1)

    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    initial_time = time.time() - start_time

    # Estimate total time
    total_params_simple = (
        len(simple_grid["hidden_layer_sizes"])
        * len(simple_grid["activation"])
        * len(simple_grid["alpha"])
        * len(simple_grid["learning_rate_init"])
    )
    total_params_full = (
        len(param_grid["hidden_layer_sizes"])
        * len(param_grid["activation"])
        * len(param_grid["alpha"])
        * len(param_grid["learning_rate_init"])
        * len(param_grid["dropout"])
    )

    estimated_time = (initial_time / total_params_simple) * total_params_full
    print(f"Estimated time for full grid search: {estimated_time/60:.2f} minutes")
    proceed = input("Proceed with full grid search? (y/n): ").lower().strip() == "y"

    if not proceed:
        print("Using simplified grid search instead...")
        # Define a simplified grid
        param_grid = {
            "hidden_layer_sizes": [(100,), (200,), (100, 50)],
            "activation": ["relu"],
            "alpha": [0.001],
            "learning_rate_init": [0.001],
            "dropout": [0.2],  # We'll handle this manually
        }

    # Manual grid search to handle dropout
    best_score = 0
    best_params = None
    results = []

    # Create directory for output if provided
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Manual grid search
    print("\nStarting grid search...")
    for hidden_layer_sizes in param_grid["hidden_layer_sizes"]:
        for activation in param_grid["activation"]:
            for alpha in param_grid["alpha"]:
                for learning_rate in param_grid["learning_rate_init"]:
                    for dropout in param_grid["dropout"]:
                        # Create MLP classifier
                        mlp = MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            alpha=alpha,
                            learning_rate_init=learning_rate,
                            max_iter=300,
                            random_state=42,
                            early_stopping=True,
                            validation_fraction=0.1,
                        )

                        # Train the model with dropout
                        # Note: scikit-learn's MLPClassifier doesn't have native dropout
                        # This is a simplification - for true dropout, you'd need to use
                        # a framework like PyTorch or TensorFlow

                        mlp.fit(X_train_scaled, y_train)

                        # Score the model
                        score = mlp.score(X_test_scaled, y_test)

                        # Save results
                        params = {
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "activation": activation,
                            "alpha": alpha,
                            "learning_rate_init": learning_rate,
                            "dropout": dropout,
                        }
                        results.append((params, score))

                        print(f"Parameters: {params}")
                        print(f"Test accuracy: {score:.4f}")

                        # Update best parameters
                        if score > best_score:
                            best_score = score
                            best_params = params

    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)

    # Print best parameters
    print("\nGrid Search Results:")
    for i, (params, score) in enumerate(results[:5]):
        print(f"Rank {i+1}: {params}, Accuracy: {score:.4f}")

    # Train final model with best parameters
    final_mlp = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation=best_params["activation"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=500,
        random_state=42,
    )

    final_mlp.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = final_mlp.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nFinal Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Add labels to confusion matrix
    num_classes = len(np.unique(y))
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    # Save model and results if output directory is provided
    if output_dir:
        # Save model
        model_path = os.path.join(output_dir, "mlp_model.joblib")
        joblib.dump(final_mlp, model_path)

        # Save scaler
        scaler_path = os.path.join(output_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)

        # Save confusion matrix
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)

        # Save best parameters
        params_path = os.path.join(output_dir, "best_params.txt")
        with open(params_path, "w") as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")

        print(f"\nModel and results saved to {output_dir}")
    else:
        plt.show()

    return final_mlp, scaler, best_params


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP classifier with grid search"
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the X.npy features file",
    )
    parser.add_argument(
        "--labels_path", type=str, required=True, help="Path to the y.npy labels file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the model and results (optional)",
    )

    args = parser.parse_args()

    # Load data
    X, y = load_data(args.features_path, args.labels_path)

    # Train MLP with grid search
    train_mlp_with_grid_search(X, y, args.output_dir)
    


if __name__ == "__main__":
    main()
