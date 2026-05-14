# this program recognizes activities
from preprocessing import create_feature_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
SEED = 42

# load features
#feature_df = create_feature_dataset("data")
feature_df = create_feature_dataset("data_without_no_gyro")

# split features / labels
X = feature_df.drop(columns=["activity", "person"])
y = feature_df["activity"]
person = feature_df["person"]

# encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# random split train/test
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )

# group split train/test (no person overlap)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state= SEED)

train_idx, test_idx = next(gss.split(X, y, groups=person))
print( "Train index:", person.iloc[train_idx], "Test index:", person.iloc[test_idx])

X_train, X_test = X.take(train_idx), X.take(test_idx)
y_train, y_test = y[train_idx], y[test_idx]

# normalize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train classifier
model = SVC(kernel="rbf", C=1.0, gamma="scale")

model.fit(X_train, y_train)

# evaluate
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# 1. Get predictions from the model
y_pred = model.predict(X_test)

# 2. Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 3. Display it visually
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=label_encoder.classes_
)

# Plot and show
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix Seed:" + str(SEED))
plt.show()

# Optional: Print the raw matrix to the console
print("\nConfusion Matrix:")
print(cm)







def evaluate_model(X, y, persons, seed):
    # Use GroupShuffleSplit to respect person-based separation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=persons))
    
    X_train, X_test = X.take(train_idx), X.take(test_idx)
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Scaling is crucial for SVC
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = SVC(kernel="rbf", C=1.0)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def test_several_models():
    # Test over 10 different seeds
    seeds = [5, 10, 42, 100, 1337, 7, 21, 99, 88, 123]
    results = []

    for s in seeds:
        acc = evaluate_model(X, y, person, s)
        results.append(acc)

    print(f"Mean Accuracy: {np.mean(results):.4f}")
    print(f"Std Dev: {np.std(results):.4f}")




