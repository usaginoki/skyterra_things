import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Load the data
X = np.load("siglip/X.npy")
y = np.load("siglip/y.npy")

# Set hyperparameters from best_params.txt
hidden_layer_sizes = (200,)
activation = "tanh"
alpha = 0.0001
learning_rate_init = 0.001
# Note: sklearn's MLPClassifier doesn't support dropout directly
# We'll still use the other parameters

# Create model with best parameters
model = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    alpha=alpha,
    learning_rate_init=learning_rate_init,
    max_iter=1000,
    early_stopping=True,
    random_state=42,
)

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
fold = 1

results_dir = "siglip_mlp_results/cv_models"
# Create directory for results if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

print("Starting 5-fold cross-validation...")

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=1000,
        early_stopping=True,
        random_state=42,
    )

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

    print(f"Fold {fold} Accuracy: {score:.4f}")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler for this fold
    joblib.dump(model, f"siglip_mlp_results/cv_models/mlp_fold_{fold}.joblib")
    joblib.dump(scaler, f"siglip_mlp_results/cv_models/scaler_fold_{fold}.joblib")

    fold += 1

# Print average score
print(f"Average accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Save the cross-validation results
with open("siglip_mlp_results/cv_results.txt", "w") as f:
    f.write(f"5-Fold Cross-Validation Results\n")
    f.write(f"-----------------------------\n")
    for i, score in enumerate(scores, 1):
        f.write(f"Fold {i} Accuracy: {score:.4f}\n")
    f.write(f"\nAverage accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")
