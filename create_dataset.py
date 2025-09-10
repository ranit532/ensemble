
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42,
    weights=[0.8, 0.2]  # Imbalanced dataset
)

# Create a DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['churn'] = y

# Add some categorical features
df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
df['subscription_plan'] = np.random.choice(['Basic', 'Premium', 'Standard'], size=len(df))

# Save the dataset to a CSV file
df.to_csv('churn_dataset.csv', index=False)

print("Dataset created successfully: churn_dataset.csv")
