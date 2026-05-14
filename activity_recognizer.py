# this program recognizes activities
from preprocessing import create_feature_dataset

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# load features
feature_df = create_feature_dataset("data")
#feature_df = create_feature_dataset("data_without_no_gyro")

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
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

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