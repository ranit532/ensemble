
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the dataset
df = pd.read_csv('churn_dataset.csv')

# Separate features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Identify categorical and numerical features
categorical_features = ['gender', 'subscription_plan']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# --- Stacking Classifier ---
def train_stacking(preprocessor, estimators, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name='Stacking Classifier') as run:
        # Create the StackingClassifier
        stacking_classifier = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )

        # Create the full pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', stacking_classifier)])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Log parameters and metrics
        mlflow.log_param("estimators", [est[0] for est in estimators])
        mlflow.log_param("final_estimator", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))

        # Log the model
        mlflow.sklearn.log_model(pipeline, "stacking_model")

        # Save the model for serving
        joblib.dump(pipeline, 'models/stacking_model.joblib')

        # Create and save plots
        create_plots(y_test, y_pred, y_pred_proba, 'stacking')

# --- Blending Classifier ---
def train_blending(preprocessor, estimators, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name='Blending Classifier') as run:
        # Split the training data into a sub-training set and a validation set
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42, stratify=y_train)

        # Preprocess the data
        X_train_sub_transformed = preprocessor.fit_transform(X_train_sub)
        X_val_transformed = preprocessor.transform(X_val)
        X_test_transformed = preprocessor.transform(X_test)

        # Train the base models on the sub-training set
        base_models = [est[1] for est in estimators]
        for model in base_models:
            model.fit(X_train_sub_transformed, y_train_sub)

        # Get the predictions from the base models on the validation set
        val_predictions = [model.predict_proba(X_val_transformed)[:, 1] for model in base_models]
        val_predictions = np.column_stack(val_predictions)

        # Train the meta-model on the validation predictions
        meta_model = LogisticRegression()
        meta_model.fit(val_predictions, y_val)

        # Get the predictions from the base models on the test set
        test_predictions = [model.predict_proba(X_test_transformed)[:, 1] for model in base_models]
        test_predictions = np.column_stack(test_predictions)

        # Make the final predictions using the meta-model
        y_pred_proba = meta_model.predict_proba(test_predictions)[:, 1]
        y_pred = meta_model.predict(test_predictions)

        # Log parameters and metrics
        mlflow.log_param("estimators", [est[0] for est in estimators])
        mlflow.log_param("meta_estimator", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))

        # Log the model (Note: Blending is a manual process, so we log the meta-model)
        mlflow.sklearn.log_model(meta_model, "blending_model")

        # Save the model for serving
        joblib.dump(meta_model, 'models/blending_model.joblib')

        # Create and save plots
        create_plots(y_test, y_pred, y_pred_proba, 'blending')

# --- Voting Classifier ---
def train_voting(preprocessor, estimators, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name='Voting Classifier') as run:
        # Create the VotingClassifier
        voting_classifier = VotingClassifier(estimators=estimators, voting='soft')

        # Create the full pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', voting_classifier)])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Log parameters and metrics
        mlflow.log_param("estimators", [est[0] for est in estimators])
        mlflow.log_param("voting", "soft")
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))

        # Log the model
        mlflow.sklearn.log_model(pipeline, "voting_model")

        # Save the model for serving
        joblib.dump(pipeline, 'models/voting_model.joblib')

        # Create and save plots
        create_plots(y_test, y_pred, y_pred_proba, 'voting')

# --- Plotting Function ---
def create_plots(y_test, y_pred, y_pred_proba, model_name):
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name.capitalize()} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name.capitalize()} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'plots/{model_name}_roc_curve.png')
    plt.close()

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Train the models
    train_stacking(preprocessor, estimators, X_train, y_train, X_test, y_test)
    train_blending(preprocessor, estimators, X_train, y_train, X_test, y_test)
    train_voting(preprocessor, estimators, X_train, y_train, X_test, y_test)

    print("Training complete. Check the MLflow UI at http://127.0.0.1:5000")
