import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load data from the SQL database
def load_data(sql_file):
    """
    This function loads data from the given SQL file.
    """
    # Connect to SQLite in-memory database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Load SQL script from file and execute it
    with open(sql_file, 'r') as file:
        sql_script = file.read()
    
    # Execute SQL script
    cursor.executescript(sql_script)
    
    # Load the data into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM patient", conn)
    
    # Close the connection
    conn.close()
    
    return df

# Step 2: Preprocess the data
def preprocess_data(df):
    """
    This function preprocesses the dataset by handling missing values, scaling numeric features,
    and encoding categorical features. It also splits the data into training and test sets.
    """
    # Separate features and target
    X = df.drop('Readmission30Days', axis=1)
    y = df['Readmission30Days']

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Preprocessing pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
        ('scaler', StandardScaler())  # Scale the data
    ])

    # Preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor

# Step 3: Train a machine learning model
def train_model(X_train, y_train):
    """
    This function trains a RandomForestClassifier on the preprocessed training data.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    This function evaluates the trained model by generating predictions and calculating the
    confusion matrix and classification report.
    """
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    return conf_matrix

# Step 5: Visualize the confusion matrix
def visualize_confusion_matrix(conf_matrix):
    """
    This function visualizes the confusion matrix using seaborn.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Step 6: Visualize feature importance
def visualize_feature_importance(model, feature_names):
    """
    This function visualizes feature importance for models that provide this attribute.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not support feature importance.")


def get_feature_names(preprocessor, df):
    """
    Get feature names after preprocessing (handling numeric and one-hot encoded features).
    """
    # Numeric feature names
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Categorical feature names (after one-hot encoding)
    categorical_transformer = preprocessor.named_transformers_['cat']
    if hasattr(categorical_transformer, 'named_steps') and 'onehot' in categorical_transformer.named_steps:
        onehot = categorical_transformer.named_steps['onehot']
        categorical_feature_names = onehot.get_feature_names_out(df.select_dtypes(include=['object']).columns)
    else:
        categorical_feature_names = df.select_dtypes(include=['object']).columns.tolist()
    
    # Combine numeric and categorical feature names
    return numeric_features + list(categorical_feature_names)

def visualize_feature_importance(model, feature_names):
    """
    This function visualizes feature importance for models that provide this attribute.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not support feature importance.")
# Main function to execute the ML pipeline
def main():
    """
    Main function to run the ML pipeline.
    """
    # Step 1: Load the data from the SQL file
    sql_file = '/Users/gokulakrishnan/Desktop/patient readmission/patient.sql'  # Provide the path to the uploaded SQL file
    df = load_data(sql_file)
    
    # Display the first few rows of the data
    print("Data Loaded:")
    print(df.head())

    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Step 3: Train the model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate the model
    conf_matrix = evaluate_model(model, X_test, y_test)
    
    # Step 5: Visualize the confusion matrix
    visualize_confusion_matrix(conf_matrix)
    
    # Step 6: Visualize feature importance
    feature_names = get_feature_names(preprocessor, df)  # Get updated feature names
    visualize_feature_importance(model, feature_names)
    

# Execute the program
if __name__ == "__main__":
    main()
