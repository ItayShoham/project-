# import h5py
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# def load_dataset(filename):
#     with h5py.File(filename, 'r') as hf:
#         features = np.array(hf.get('features'))
#         labels = np.array(hf.get('labels'))
#     return features, labels

# def train_model(features, labels):
#     # Splits dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    
#     clf = RandomForestClassifier(random_state=42)

#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }

#     # Initializes GridSearchCV
#     grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

#     # Train the model using grid search
#     grid_search.fit(X_train, y_train)

#     print(f"Best parameters: {grid_search.best_params_}")

#     # Use the best estimator from the grid search to make predictions
#     best_clf = grid_search.best_estimator_
#     y_pred = best_clf.predict(X_test)

#     # Evaluates the model
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)

#     # Prints the results
#     print(f"Accuracy on Test Set: {accuracy:.2f}")
#     print("Classification Report:")
#     print(report)

#     return best_clf

# def main():
#     dataset_filename = 'connect4_large_dataset.h5'

#     # Load dataset
#     features, labels = load_dataset(dataset_filename)

#     # Train model
#     best_model = train_model(features, labels)

#     # Optionally, save the trained model
#     # joblib.dump(best_model, 'connect4_rf_model.pkl')

# if __name__ == "__main__":
#     main()

import h5py
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

def load_dataset(filename):
    with h5py.File(filename, 'r') as hf:
        features = np.array(hf.get('features'))
        labels = np.array(hf.get('labels'))
    return features, labels

def preprocess_data(features, labels):
    # Reshape features to 2D if necessary
    features = features.reshape(features.shape[0], -1)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, labels, scaler

def train_model(features, labels):
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Define models to try
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        if model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                                           n_iter=20, cv=5, n_jobs=-1, verbose=2, random_state=42)
        
        # Train the model using random search
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        
        # Use the best estimator to make predictions
        best_estimator = random_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Perform cross-validation
        cv_scores = cross_val_score(best_estimator, features, labels, cv=5)
        
        print(f"Accuracy on Test Set: {accuracy:.2f}")
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})")
        print("Classification Report:")
        print(report)
        
        if np.mean(cv_scores) > best_score:
            best_score = np.mean(cv_scores)
            best_model = best_estimator
    
    return best_model

def main():
    dataset_filename = 'connect4_large_dataset.h5'
    
    # Load dataset
    features, labels = load_dataset(dataset_filename)
    
    # Preprocess data
    features_scaled, labels, scaler = preprocess_data(features, labels)
    
    # Train model
    best_model = train_model(features_scaled, labels)
    
    # Save the trained model and scaler
    joblib.dump(best_model, 'connect4_best_model.pkl')
    joblib.dump(scaler, 'connect4_scaler.pkl')
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    main()