# this program recognizes activities
from preprocessing import create_feature_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# load features
feature_df = create_feature_dataset("data")

# split features / labels
X = feature_df.drop(columns=["activity"])
y = feature_df["activity"]

# encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

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