from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, \
    classification_report
from collections import defaultdict

import pandas as pd
from collections import defaultdict

import pandas as pd
import pickle
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedBaggingClassifier



def Balanced_Bagging_Classifier(X_train_scaled, y_train, X_test_scaled, y_test, save_results=False):
    """
    Evaluate multiple classifiers on given data, compute CV metrics and optionally save results to a file.

    Args:
    X_train_scaled (array-like): Scaled training features.
    y_train (array-like): Training labels.
    X_test_scaled (array-like): Scaled testing features.
    y_test (array-like): Testing labels.
    save_results (bool): Flag to determine if results should be saved to file.

    Returns:
    None
    """
    results = []
    results_file = 'BalancedBaggingClassifier_first_results.pkl'

    if not save_results and os.path.exists(results_file):
        with open(results_file, 'rb') as file:
            results = pickle.load(file)
    else:
        f1_scorer = make_scorer(f1_score, average='macro')
        base_classifiers = [
            ('XGBoost', XGBClassifier(objective='multi:softprob', eval_metric='logloss', use_label_encoder=False)),
            ('Random Forest', RandomForestClassifier()),
            ('KNN', KNeighborsClassifier())
        ]

        for name, base_classifier in base_classifiers:
            bbc = BalancedBaggingClassifier(
                base_estimator=base_classifier,
                n_estimators=50,
                random_state=42
            )

            bbc.fit(X_train_scaled, y_train)

            predictions = bbc.predict(X_test_scaled)

            metrics = calculate_classification_metrics(y_test, predictions)
            
            cv_metrics = {
                'precision': cross_val_score(bbc, X_train_scaled, y_train, cv=10, scoring='precision_macro'),
                'recall': cross_val_score(bbc, X_train_scaled, y_train, cv=10, scoring='recall_macro'),
                'f1': cross_val_score(bbc, X_train_scaled, y_train, cv=10, scoring=f1_scorer)
            }

            results.append((name, metrics, cv_metrics))

        # Optionally save the results to a file
        if save_results:
            with open(results_file, 'wb') as file:
                pickle.dump(results, file)

    return results  

def Balanced_Bagging_Classifier1(X_train_scaled, y_train, X_test_scaled, y_test, save_results=False):
    """
    Evaluate multiple classifiers on given data and optionally save results to a file.

    Args:
    X_train_scaled (array-like): Scaled training features.
    y_train (array-like): Training labels.
    X_test_scaled (array-like): Scaled testing features.
    y_test (array-like): Testing labels.
    save_results (bool): Flag to determine if results should be saved to file.

    Returns:
    None
    """
    results = []
    results_file = 'BalancedBaggingClassifier_first_results.pkl'

    # Check if results should be loaded from file instead of recomputing
    if not save_results and os.path.exists(results_file):
        with open(results_file, 'rb') as file:
            results = pickle.load(file)
    else:
        base_classifiers = [
            ('XGBoost', XGBClassifier(objective='multi:softprob', eval_metric='logloss', use_label_encoder=False)),
            ('Random Forest', RandomForestClassifier()),
            ('KNN', KNeighborsClassifier())
        ]

        # Loop through each base classifier
        for name, base_classifier in base_classifiers:
            bbc = BalancedBaggingClassifier(
                base_estimator=base_classifier,
                n_estimators=50,  
                random_state=42
            )

            bbc.fit(X_train_scaled, y_train)
            predictions = bbc.predict(X_test_scaled)
            results.append((name, calculate_classification_metrics(y_test, predictions)))

        if save_results:
            with open(results_file, 'wb') as file:
                pickle.dump(results, file)

    return results  


def train_xgboost_model(X_train_scaled, y_train_encoded):
    model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')
    model.fit(X_train_scaled, y_train_encoded)
    return model

def evaluate_cross_validation(model, X_train_scaled, y_train_encoded):
    f1_scorer = make_scorer(f1_score, average='macro')
    cv_metrics = {
        'precision': cross_val_score(model, X_train_scaled, y_train_encoded, cv=10, scoring='precision_macro'),
        'recall': cross_val_score(model, X_train_scaled, y_train_encoded, cv=10, scoring='recall_macro'),
        'f1': cross_val_score(model, X_train_scaled, y_train_encoded, cv=10, scoring=f1_scorer)
    }
    return cv_metrics

def save_results(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def basic_model_and_evaluate(X_train, y_train, X_test, run_model=True):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    results= {'cv_metrics': ''}

    if run_model:
        model = train_xgboost_model(X_train, y_train_encoded)
        cv_metrics = evaluate_cross_validation(model, X_train, y_train_encoded)
        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        save_results('base_model_results.pkl', {
            'y_pred': y_pred,
            'cv_metrics': cv_metrics
        })

        for metric, scores in cv_metrics.items():
            print(f"Cross-validated Macro {metric.capitalize()}: {scores.mean():.2f} ± {scores.std():.2f}")

    else:
        results = load_results('base_model_results.pkl')
        y_pred = results['y_pred']
        for metric, scores in results['cv_metrics'].items():
            print(f"Cross-validated Macro {metric.capitalize()}: {scores.mean():.2f} ± {scores.std():.2f}")

    return y_pred, results['cv_metrics']

def xgboost_SMOTE_model_and_evaluate(X_train, y_train, X_test, run_model=True):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    results= {'cv_metrics': ''}

    if run_model:
        model = train_xgboost_model(X_train, y_train_encoded)
        cv_metrics = evaluate_cross_validation(model, X_train, y_train_encoded)
        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        save_results('model_SMOTE_results.pkl', {
            'y_pred': y_pred,
            'cv_metrics': cv_metrics
        })

        # for metric, scores in cv_metrics.items():
        #     print(f"Cross-validated Macro {metric.capitalize()}: {scores.mean():.2f} ± {scores.std():.2f}")

    else:
        results = load_results('model_SMOTE_results.pkl')
        y_pred = results['y_pred']
        # for metric, scores in results['cv_metrics'].items():
        #     print(f"Loaded Cross-validated Macro {metric.capitalize()}: {scores.mean():.2f} ± {scores.std():.2f}")

    return y_pred, results['cv_metrics']

def xgboost_model_and_evaluate(X_train, y_train, X_test):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    model = train_xgboost_model(X_train, y_train_encoded)
    cv_metrics = evaluate_cross_validation(model, X_train, y_train_encoded)
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    return y_pred, cv_metrics



def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate various classification metrics.

    Args:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    dict: A dictionary containing accuracy, macro-precision, macro-recall,
          macro-F1 score, balanced accuracy, and classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, output_dict=True)

    return {
        'Accuracy': accuracy,
        'Macro Precision': macro_precision,
        'Macro Recall': macro_recall,
        'Macro F1 Score': macro_f1,
        'Balanced Accuracy': balanced_accuracy,
        'Classification Report': classification_rep
    }



def get_classification_report_metrics(results, *args):
    data = defaultdict(list)


    for model_results in results:
        #data['Accuracy'].append(model_results['Accuracy'])
        data['Macro Precision'].append(model_results['Macro Precision'])
        data['Macro Recall'].append(model_results['Macro Recall'])
        data['Macro F1 Score'].append(model_results['Macro F1 Score'])
        data['Balanced Accuracy'].append(model_results['Balanced Accuracy'])
        #data['Classification Report'].append(model_results['Classification Report'])

    scores = pd.DataFrame(data=data, index=[*args])
    return scores





