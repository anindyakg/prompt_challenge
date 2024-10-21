import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, BaggingRegressor,
                            AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Define feature columns globally to ensure consistency
NUMERICAL_FEATURES = ['Age', 'Height', 'Weight', 'BMI']
CATEGORICAL_FEATURES = ['BMI_Category', 'Age_Bucket']
BINARY_FEATURES = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants',
                  'AnyChronicDiseases', 'KnownAllergies', 
                  'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries']

def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Calculate BMI
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    
    # Create BMI Category
    df['BMI_Category'] = pd.cut(df['BMI'], 
                               bins=[0, 18.5, 25, 30, float('inf')],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Create Age Bucket
    df['Age_Bucket'] = pd.cut(df['Age'],
                             bins=[0, 25, 35, 45, 55, float('inf')],
                             labels=['18-25', '26-35', '36-45', '46-55', '56+'])
    
    return df

def encode_and_scale_features(df):
    # Create copy of dataframe
    df_processed = df.copy()
    
    # Encode categorical variables
    le_bmi = LabelEncoder()
    le_age = LabelEncoder()
    
    df_processed['BMI_Category_encoded'] = le_bmi.fit_transform(df_processed['BMI_Category'])
    df_processed['Age_Bucket_encoded'] = le_age.fit_transform(df_processed['Age_Bucket'])
    
    # Save encoders
    joblib.dump(le_bmi, 'bmi_encoder.pkl')
    joblib.dump(le_age, 'age_encoder.pkl')
    
    # Scale numerical features
    scaler = StandardScaler()
    df_processed[NUMERICAL_FEATURES] = scaler.fit_transform(df_processed[NUMERICAL_FEATURES])
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    return df_processed

def perform_eda(df):
    # Create correlation matrix
    plt.figure(figsize=(12, 8))
    numerical_cols = NUMERICAL_FEATURES + ['PremiumPrice']
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # Distribution plots
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'{col}_distribution.png')
        plt.close()

def remove_outliers(df):
    df_clean = df.copy()
    
    for col in NUMERICAL_FEATURES:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def get_feature_columns():
    return (NUMERICAL_FEATURES + 
            ['BMI_Category_encoded', 'Age_Bucket_encoded'] + 
            BINARY_FEATURES)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Bagging': BaggingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        # Save Random Forest model
        if name == 'Random Forest':
            joblib.dump(model, 'random_forest_model.pkl')
            # Save feature columns order
            joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')
    
    return results

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('insurance_short.csv')
    
    # Perform EDA
    perform_eda(df)
    
    # Remove outliers
    df_clean = remove_outliers(df)
    
    # Encode and scale features
    df_processed = encode_and_scale_features(df_clean)
    
    # Get feature columns in correct order
    feature_cols = get_feature_columns()
    
    # Prepare features for modeling
    X = df_processed[feature_cols]
    y = df_processed['PremiumPrice']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Print results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
