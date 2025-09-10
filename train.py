import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI if you have a server, otherwise it will log locally
# mlflow.set_tracking_uri("http://localhost:5000")

# 1. Load Data
df = pd.read_csv('churn_dataset.csv')

# Define features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Identify categorical and numerical features
categorical_features = ['gender', 'subscription_plan']
numerical_features = [col for col in X.columns if col not in categorical_features]

# 2. Preprocessing
# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. Data Splitting
# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_blend, X_val_blend, y_train_blend, y_val_blend = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

# --- Utility function for evaluation and plotting ---
def evaluate_model(model, X_test, y_test, model_name, run_name):
    """Evaluates the model, logs metrics and plots to MLflow."""
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Metrics for {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

    # Log metrics to MLflow
    mlflow.log_metrics({
        f"{model_name}_accuracy": accuracy,
        f"{model_name}_precision": precision,
        f"{model_name}_recall": recall,
        f"{model_name}_f1_score": f1,
        f"{model_name}_roc_auc": roc_auc
    })

    # Create and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    cm_path = f"images/{run_name}_{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # Create and save ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    roc_path = f"images/{run_name}_{model_name}_roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # Log plots as artifacts to MLflow
    mlflow.log_artifact(cm_path, "plots")
    mlflow.log_artifact(roc_path, "plots")
    
    return model

# --- Main Training Logic ---
if __name__ == "__main__":
    
    run_name = f"ensemble_training_{int(time.time())}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("run_name", run_name)
        
        # --- 4. Blending Classifier ---
        print("\n--- Training Blending Classifier ---")
        
        # Define base models
        base_models_blend = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42))
        ]

        # Preprocess training and validation data for blending
        X_train_blend_processed = preprocessor.fit_transform(X_train_blend)
        X_val_blend_processed = preprocessor.transform(X_val_blend)

        # Train base models and create meta-features
        meta_features = []
        for name, model in base_models_blend:
            print(f"Training base model: {name}")
            model.fit(X_train_blend_processed, y_train_blend)
            val_preds = model.predict_proba(X_val_blend_processed)[:, 1]
            meta_features.append(val_preds)
        
        meta_features = np.array(meta_features).T

        # Train meta-model
        print("Training meta-model for blending...")
        meta_model_blend = LogisticRegression(random_state=42, max_iter=1000)
        meta_model_blend.fit(meta_features, y_val_blend)

        # To make predictions with the blending model, we need a custom class/pipeline
        # For simplicity in this PoC, we'll create a function to represent the pipeline
        def predict_blending(X):
            X_processed = preprocessor.transform(X)
            base_preds = [model.predict_proba(X_processed)[:, 1] for _, model in base_models_blend]
            meta_feats = np.array(base_preds).T
            return meta_model_blend.predict(meta_feats), meta_model_blend.predict_proba(meta_feats)

        # We need to wrap this logic for evaluation
        class BlendingWrapper:
            def predict(self, X):
                preds, _ = predict_blending(X)
                return preds
            def predict_proba(self, X):
                _, probas = predict_blending(X)
                return probas

        blending_model = BlendingWrapper()
        
        # Evaluate the blending model
        evaluate_model(blending_model, X_test, y_test, "Blending", run_name)
        
        # Note: Saving the blending model requires custom logic to save all base models and the meta-model.
        # For this PoC, we will focus on saving the StackingClassifier which is more straightforward.

        # --- 5. Stacking Classifier ---
        print("\n--- Training Stacking Classifier ---")
        
        # Define base models for stacking
        base_models_stack = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=42, n_estimators=50)),
        ]

        # Define the meta-model for stacking
        meta_model_stack = GradientBoostingClassifier(random_state=42)

        # Create the Stacking Classifier pipeline
        stacking_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', StackingClassifier(
                estimators=base_models_stack,
                final_estimator=meta_model_stack,
                cv=StratifiedKFold(n_splits=5)
            ))
        ])

        # Train the stacking model
        print("Training StackingClassifier...")
        stacking_pipeline.fit(X_train, y_train)

        # Evaluate the stacking model
        evaluate_model(stacking_pipeline, X_test, y_test, "Stacking", run_name)

        # Save the trained stacking model
        model_path = f"models/{run_name}_stacking_model.joblib"
        joblib.dump(stacking_pipeline, model_path)
        print(f"Stacking model saved to {model_path}")

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=stacking_pipeline,
            artifact_path="stacking_model",
            registered_model_name="EnsembleStackingClassifier"
        )
        
        print("\n--- Training complete! ---")
        print(f"To see the results, run: mlflow ui")
        print(f"Find your experiment run named: {run_name}")