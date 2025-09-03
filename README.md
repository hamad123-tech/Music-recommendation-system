# Music-recommendation-system
A simple music recommendation system using Python, Random Forest, and a simulated dataset.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import random
from datetime import datetime, timedelta

# --------------------
# 1. Simulate Dummy Dataset
# --------------------
num_users = 50
num_songs = 100
num_records = 2000  # total user-song interactions

np.random.seed(42)

# Generate users
users = np.random.randint(1, num_users+1, num_records)

# Generate fake song titles
base_titles = [
    "Midnight Beats", "Love in the Rain", "Ocean Dreams", "Golden Sunrise", "Shadow Dance",
    "Electric Nights", "Whispering Stars", "Moonlit Roads", "Firefly Glow", "Echoes of You",
    "Crystal Skies", "Velvet Storm", "Lost in Time", "Chasing Horizons", "Fading Echoes",
    "Starlight Symphony", "Dreamcatcher", "Neon Pulse", "Silver Lining", "Mystic River"
]

# Create 100 unique titles (adding suffixes if needed)
song_titles = [f"{random.choice(base_titles)} #{i}" for i in range(1, num_songs+1)]
songs = np.random.choice(song_titles, num_records)

# Generate timestamps (last 60 days)
base_date = datetime.today()
timestamps = [base_date - timedelta(days=np.random.randint(0, 60)) for _ in range(num_records)]

# Simulate target (1 = repeated within a month, 0 otherwise)
targets = np.random.choice([0, 1], size=num_records, p=[0.7, 0.3])

# Create DataFrame
data = pd.DataFrame({
    "user_id": users,
    "song_id": songs,
    "timestamp": timestamps,
    "target": targets
})

print("Sample Data:\n", data.head())

# --------------------
# 2. Preprocessing
# --------------------
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['dayofweek'] = data['timestamp'].dt.dayofweek
data['hour'] = data['timestamp'].dt.hour

# Features and Target
X = data[['user_id', 'song_id', 'dayofweek', 'hour']]
y = data['target']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['user_id', 'song_id'])

# --------------------
# 3. Train/Test Split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------
# 4. Model Training
# --------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --------------------
# 5. Evaluation
# --------------------
y_pred = model.predict(X_test)
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------
# 6. Recommendation Function
# --------------------
def recommend_songs(user_id, model, X, top_n=5):
    """
    Recommend top_n songs for a given user based on model predictions.
    """
    user_cols = [col for col in X.columns if col.startswith('user_id_')]
    song_cols = [col for col in X.columns if col.startswith('song_id_')]
    
    if f"user_id_{user_id}" not in X.columns:
        print("User not found in dataset.")
        return []
    
    user_data = X.copy()
    user_data = user_data[user_data[f"user_id_{user_id}"] == 1]
    
    if user_data.empty:
        print("No history found for this user.")
        return []
    
    preds = model.predict_proba(user_data)[:, 1]
    user_data = user_data.copy()
    user_data['score'] = preds
    
    top_songs = (
        user_data.sort_values('score', ascending=False)
        .head(top_n)
        .loc[:, song_cols]
    )
    
    recommended = []
    for row in top_songs.index:
        song_id = [col.replace("song_id_", "") for col in song_cols if user_data.loc[row, col] == 1]
        if song_id:
            recommended.append(song_id[0])
    
    return recommended

# --------------------
# 7. Example Usage
# --------------------
user_example = random.randint(1, num_users)  # pick random user
print(f"\nRecommended songs for user {user_example}:")
print(recommend_songs(user_example, model, X_test, top_n=5))
