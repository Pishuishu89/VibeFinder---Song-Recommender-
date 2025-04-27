import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# --- Load dataset safely ---
@st.cache_data(show_spinner=True)
def load_data():
    csv_url = 'https://drive.google.com/uc?id=1FY62kEf3YE82QrBLeidFGmA8ezLEryyx'
    try:
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Streamlit App ---
st.title('🎵 VibeFinder — Song Recommender')

# Load the data
df = load_data()

if df is None:
    st.stop()

# Features we use
features = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'speechiness',
    'tempo', 'valence'
]

# Load pre-trained models
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

# Recalculate clusters for the dataset
X = df[features]
X_scaled = scaler.transform(X)
df['cluster'] = kmeans.predict(X_scaled)

# Recommendation function
def recommend_song(song_name, data, scaler, kmeans_model, features, n_recs=5):
    song = data[data['track_name'].str.lower() == song_name.lower()]
    if song.empty:
        return None
    song_features = scaler.transform(song[features])
    cluster_label = kmeans_model.predict(song_features)[0]
    recommendations = data[data['cluster'] == cluster_label]
    recommendations = recommendations.sample(n=min(n_recs, len(recommendations)))
    return recommendations[['track_name', 'artist_name']]

# --- Frontend ---
song_input = st.text_input('Enter a song name you like:')

if st.button('Recommend'):
    if song_input:
        recs = recommend_song(song_input, df, scaler, kmeans, features)
        if recs is not None:
            st.success('Here are some similar songs:')
            st.table(recs)
        else:
            st.error('Song not found! Try typing a different song name.')
