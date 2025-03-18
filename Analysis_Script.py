# Import necessary libraries
import pandas as pd
import numpy as np
import GEOparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif


# Step 1: Load gene expression data
def load_data(file_path):
    """
    Load gene expression data from a CSV file.
    - Rows: Samples
    - Columns: Genes (features) + one column for labels (e.g., cancer vs. normal)
    """
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")
    return data


# Step 2: Preprocess the data
def preprocess_data(data):
    """
    Preprocess the gene expression data:
    - Separate features (genes) and labels.
    - Normalize the data.
    - Perform feature selection.
    """
    # Separate features (X) and labels (y)
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]  # Last column is the label

    # Normalize the data (mean=0, variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform feature selection (select top 10 genes)
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_scaled, y)

    print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}.")
    return X_selected, y


# Step 3: Train a machine learning model
def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# Main function
def main():
    # Step 1: Load the data
    file_path = "gene_expression_data.csv"  # Replace with your dataset path
    data = load_data(file_path)

    # Step 2: Preprocess the data
    X, y = preprocess_data(data)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train the model
    print("Training the model...")
    model = train_model(X_train, y_train)

    # Step 5: Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
