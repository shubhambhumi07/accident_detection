import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('sensor_data.csv')

# Encode categorical features
road_encoder = LabelEncoder()
df['road_condition_encoded'] = road_encoder.fit_transform(df['road_condition'])

# Save encoder for later use
joblib.dump(road_encoder, 'road_condition_encoder.pkl')

# -------------------------
# 1. Train Accident Classifier
# -------------------------
X_accident = df[['speed', 'acceleration', 'impact_force', 'road_condition_encoded']]
y_accident = df['is_accident']

X_train, X_test, y_train, y_test = train_test_split(X_accident, y_accident, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("🔍 Accident Classifier Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'accident_classifier.pkl')

# -------------------------
# 2. Train Severity Predictor
# -------------------------
# Filter only rows with accidents for severity prediction
df_sev = df[df['is_accident'] == 1].copy()

# Encode severity
severity_encoder = LabelEncoder()
df_sev['severity_encoded'] = severity_encoder.fit_transform(df_sev['severity'])

X_severity = df_sev[['speed', 'acceleration', 'impact_force', 'road_condition_encoded']]
y_severity = df_sev['severity_encoded']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_severity, y_severity, test_size=0.2, random_state=42)

severity_model = RandomForestClassifier(n_estimators=100, random_state=42)
severity_model.fit(X_train_s, y_train_s)

# Evaluate
y_pred_s = severity_model.predict(X_test_s)
print("🔍 Severity Predictor Report:\n", classification_report(y_test_s, y_pred_s))

# Save severity model and encoder
joblib.dump(severity_model, 'severity_model.pkl')
joblib.dump(severity_encoder, 'severity_encoder.pkl')

print("✅ Models trained and saved: accident_classifier.pkl, severity_model.pkl")
