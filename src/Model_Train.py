import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # <--- ADDED METRICS
import matplotlib.pyplot as plt # <--- ADDED FOR GRAPHING
import seaborn as sns           # <--- ADDED FOR STYLING
import pickle

print("1. Loading dataset...")
try:
    df = pd.read_csv('dataset1.csv')
except FileNotFoundError:
    print("ERROR: dataset.csv not found! Make sure it is in the same folder.")
    exit()

X = df.drop('label', axis=1) 
y = df['label']              

print(f"Loaded {len(df)} total frames of hand signs.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n2. Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate the Model ────────────────────────────────────────────
print("\n3. Evaluating model on hidden test data...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

# ── 🌟 NEW: Detailed Metrics & Visualizations ────────────────────────
print("\nGenerating Detailed Classification Report...")
# This prints a text report showing precision and recall for EVERY letter
print(classification_report(y_test, y_pred))

print("Generating Confusion Matrix Graph...")
# Generate the matrix data
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Create a visual figure
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)

plt.title('Sign Language Model Confusion Matrix')
plt.ylabel('Actual Letter')
plt.xlabel('Predicted Letter')

# Save the graph as an image file so you can view it later
plt.savefig('confusion_matrix.png', bbox_inches='tight')
print("-> Saved visual graph as 'confusion_matrix.png'")
# plt.show() # Uncomment this if you want the window to pop up on your screen

# ── Export the Model ──────────────────────────────────────────────
model_filename = 'sign_lang_model.pkl'
print(f"\n4. Exporting trained model to {model_filename}...")
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print("SUCCESS! Your model is ready for deployment.")