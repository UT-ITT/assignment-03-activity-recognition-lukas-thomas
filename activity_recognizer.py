import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

# Assuming preprocessing.py exists in the same directory
from preprocessing import create_feature_dataset

# Constants
SEED = 1000
DATA_PATH = "data"

SEEDS_FOR_TESTING = [5, 10, 42, 100, 420, 1000, 21, 99, 88, 69]

def load_and_preprocess_data(path, window_size=250, step_size=125):
    """Loads dataset and encodes labels."""
    df = create_feature_dataset(path)
    
    X = df.drop(columns=["activity", "person"])
    y_raw = df["activity"]
    persons = df["person"]
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    return X, y, persons, label_encoder

def get_pipeline():
    """Creates a pipeline that scales data and then applies the SVM."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel="rbf", C=1, gamma="scale"))
    ])


def plot_cm(y_test, y_pred, target_names, seed):
    """Generates and displays a confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix (Seed: {seed})")
    plt.show()
    return cm

def run_single_evaluation(X, y, persons, seed, label_encoder, show_plot=True, classifier=get_pipeline()):
    """Performs one train/test split and evaluates the model."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=persons))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Build and train pipeline
    clf = classifier
    clf.fit(X_train, y_train)
    
    # Evaluate
    accuracy = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    
    print(f"--- Results for Seed {seed} ---")
    print(f"Accuracy: {accuracy:.4f}")

    if show_plot:
        plot_cm(y_test, y_pred, label_encoder.classes_, seed)
        
    return clf, accuracy

def run_multi_seed_test(X, y, persons, seeds):
    """Tests model stability across multiple random seeds."""
    print(f"\nEvaluating over {len(seeds)} seeds...")
    results = []
    for s in seeds:
        # Run without showing plots for brevity
        clf, acc = run_single_evaluation(X, y, persons, s, None, show_plot=False)
        results.append(acc)
    
    print(f"Mean Accuracy: {np.mean(results):.4f}")
    print(f"Std Dev: {np.std(results):.4f}")



def run_logo_evaluation(X, y, persons, label_encoder, classifier = get_pipeline()):
    logo = LeaveOneGroupOut()
    scores = []
    
    # logo.split returns indices for one person at a time
    for train_idx, test_idx in logo.split(X, y, groups=persons):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        classifier.fit(X_train, y_train)
        
        score = classifier.score(X_test, y_test)
        scores.append(score)
        
        # Identify which person was tested (optional)
        tested_person = persons.iloc[test_idx].unique()[0]
        print(f"Accuracy for Person {tested_person}: {score:.4f}")
    
    print(f"\nOverall LOGO Mean Accuracy: {np.mean(scores):.4f}")

# Helper function to train the classifier and return it for use in the main loop
def train_classifier():
    print("Training classifier for real-time prediction...")
    print("Please wait, this may take a moment...")
    X, y, persons, encoder = load_and_preprocess_data(DATA_PATH)
    clf = get_pipeline()
    clf, accuracy = run_single_evaluation(X, y, persons, 42, encoder,False, clf)

    clf.label_encoder = encoder  # Attach encoder to the classifier for later use
    clf.feature_columns = X.columns  # Attach feature column names for later use

    return clf

if __name__ == "__main__":
    # 1. Prepare Data
    print("Loading data...")
    X, y, persons, encoder = load_and_preprocess_data(DATA_PATH)
    
    # 2. Single Run (with Plotting)
    run_single_evaluation(X, y, persons, SEED, encoder)
    
    # 3. Robustness Check (Multiple Seeds)
    run_multi_seed_test(X, y, persons, SEEDS_FOR_TESTING)

    # 4 LOGO 
    run_logo_evaluation(X, y, persons, encoder)

    # Test for a specific classifier object on test features
    # clf1 = train_classifier()
    # run_logo_evaluation(X, y, persons, encoder, clf1)
    