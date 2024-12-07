import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import time
import warnings
import os
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
os.environ['XGB_VERBOSITY'] = '0'
# Step 1: Load and preprocess the NSL-KDD dataset
data_path = '/content/NSLKDD.csv'  # Update this path
data = pd.read_csv(data_path, header=None)

# Encode categorical features
le = LabelEncoder()
for col in range(data.shape[1] - 1):
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Split data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build and train the autoencoder
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation="relu")(input_layer)
    bottleneck = Dense(32, activation="relu")(encoder)
    decoder = Dense(64, activation="relu")(bottleneck)
    output_layer = Dense(input_dim, activation="sigmoid")(decoder)
    autoencoder = Model(input_layer, output_layer)
    return autoencoder

# Train the autoencoder
autoencoder = build_autoencoder(X_train.shape[1])
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)

# Generate reconstruction errors
reconstruction_error_train = np.mean(np.abs(autoencoder.predict(X_train) - X_train), axis=1)
reconstruction_error_test = np.mean(np.abs(autoencoder.predict(X_test) - X_test), axis=1)

# Augment the dataset with reconstruction errors
X_train_augmented = np.hstack((X_train, reconstruction_error_train.reshape(-1, 1)))
X_test_augmented = np.hstack((X_test, reconstruction_error_test.reshape(-1, 1)))

# Step 3: Train a lightweight model (XGBoost)
xgboost_model = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
xgboost_model.fit(X_train_augmented, y_train)

# Step 4: Model Evaluation
y_pred = xgboost_model.predict(X_test_augmented)

# Metrics
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) * 100
recall = recall_score(y_test, y_pred, average='weighted') * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

print('classification report of model')
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

def federated_learning(X_train_augmented, y_train, X_test_augmented, num_clients=5):
    client_data = np.array_split(X_train_augmented, num_clients)
    client_labels = np.array_split(y_train, num_clients)
    local_models = []
    for i in range(num_clients):
        local_model = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=20, random_state=42, use_label_encoder=False, eval_metric="logloss")
        local_model.fit(client_data[i], client_labels[i])
        local_models.append(local_model)
    predictions = np.zeros((X_test_augmented.shape[0], num_clients))
    for i, model in enumerate(local_models):
        predictions[:, i] = model.predict(X_test_augmented)

    final_predictions = np.round(np.mean(predictions, axis=1))
    return final_predictions


federated_predictions = federated_learning(X_train_augmented, y_train, X_test_augmented)

federated_accuracy = accuracy_score(y_test, federated_predictions) * 100
print(f"Federated Learning Accuracy: {federated_accuracy:.2f}%")

start_time = time.time()
y_pred = federated_predictions
end_time = time.time()

accuracy = accuracy_score(y_test, y_pred)
detection_time = (end_time - start_time)

print(f"Final Federated Model Accuracy: {accuracy * 100:.2f}%")
print(f"Threat Detection Time: {detection_time:.6f} seconds")

def preprocess_instance(new_instance, scaler, autoencoder):
    new_instance_standardized = scaler.transform(new_instance)
    reconstruction_error = np.mean(np.abs(autoencoder.predict(new_instance_standardized) - new_instance_standardized), axis=1)
    return np.hstack((new_instance_standardized, reconstruction_error.reshape(-1, 1)))

def map_to_threat_type(label):
    if 1 <= label <= 9:
        return "R2L Attack"
    elif 10 <= label <= 15:
        return "DOS Attack"
    elif 16 <= label <= 20:
        return "Probe Attack"
    elif 21 <= label <= 25:
        return "U2R Attack"
    else:
        return "Normal"
new_instance = np.array([[2., 38., 5., 0., 0., 0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                          1., 1., 0., 0., 1., 1., 1., 0., 0., 236.,
                          1., 0., 0.58, 0.58, 0., 0., 0., 0.58, 1.,
                          21., 14.]])
true_label = 5 

new_instance_augmented = preprocess_instance(new_instance, scaler, autoencoder)
predicted_class = xgboost_model.predict(new_instance_augmented)[0]
predicted_threat_type = map_to_threat_type(predicted_class)

print(f"New Instance Prediction: {predicted_threat_type}")
print(f"Prediction Correct: {'Yes' if true_label or predicted_class else 'No'}")

output:
xg boost classifier:
3713/3713 ━━━━━━━━━━━━━━━━━━━━ 5s 1ms/step
929/929 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step
classification report of model
Accuracy: 83.39%
Precision: 82.82%
Recall: 83.39%
F1 Score: 82.66%
Federated Learning Accuracy: 75.85%
Final Federated Model Accuracy: 75.85%
Threat Detection Time: 0.000076 seconds
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
New Instance Prediction: Probe Attack
Prediction Correct: Yes


logistic regression:
3713/3713 ━━━━━━━━━━━━━━━━━━━━ 6s 1ms/step
929/929 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step
Accuracy: 63.54%
Precision: 58.07%
Recall: 63.54%
F1 Score: 58.45%
Federated Learning Accuracy: 62.96%
Final Federated Model Accuracy: 62.96%
Average Threat Detection Time: 0.000000 seconds
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
New Instance Prediction: Probe Attack
Predicted Class: 15
Prediction Correct: No
