import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# --- Load everything ---
st.title('🎵 VibeFinder — Song Recommender')

# Show loading spinner while loading CSV
with st.spinner('Loading song database...'):
    # Load dataset from Google Drive
    csv_url = 'https://drive.google.com/uc?id=1FY62kEf3YE82QrBLeidFGmA8ezLEryyx'
    df = pd.read_csv(csv_url)

# Features we use
features = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'speechiness',
    'tempo', 'valence'
]

# Load pre-trained scaler and kmeans model
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

# Recalculate clusters for the dataset
X = df[features]
X_scaled = scaler.transform(X)
df['cluster'] = kmeans.predict(X_scaled)

# --- Recommendation Function ---
def recommend_song(song_name, data, scaler, kmeans_model, features, n_recs=5):
    song = data[data['track_name'].str.lower() == song_name.lower()]
    if song.empty:
        return None
    song_features = scaler.transform(song[features])
    cluster_label = kmeans_model.predict(song_features)[0]
    recommendations = data[data['cluster'] == cluster_label]
    recommendations = recommendations.sample(n=min(n_recs, len(recommendations)))
    return recommendations[['track_name', 'artist_name']]

# --- User Interface ---
song_input = st.text_input('Enter a song name you like:')

if st.button('Recommend'):
    if song_input:
        recs = recommend_song(song_input, df, scaler, kmeans, features)
        if recs is not None:
            st.success('Here are some similar songs:')
            st.table(recs)
        else:
            st.error('Song not found! Try typing a different song name.')

