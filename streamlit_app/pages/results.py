"""
Results and recommendations display page
"""
import streamlit as st
import pandas as pd
from src.utils import get_movie_poster_url, format_genres, get_recommendation_explanation
from src.hybrid_recommender import HybridRecommender
import os

# Page configuration
st.set_page_config(
    page_title="Recommendations - MovieLens Recommender",
    page_icon="🎬",
    layout="wide"
)

# Title and introduction
st.title("🎬 Your Personalized Recommendations")
st.markdown("---")

# Check if user has preferences
if 'user_preferences' not in st.session_state:
    st.warning("⚠️ Please fill out the questionnaire first!")
    st.stop()

# Initialize recommender if not already done
if 'recommender' not in st.session_state:
    st.session_state.recommender = HybridRecommender()
    st.session_state.recommender.train()

# Get preferences
preferences = st.session_state.user_preferences
ratings = preferences['ratings']
genres = preferences['genres']
time_preference = preferences['time_preference']
discovery_type = preferences['discovery_type']

# Charger la correspondance id -> titre
movies_df = pd.read_csv("data/processed/movies_clean.csv")
movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['title']))

# Generate recommendations
with st.spinner("🎯 Generating your personalized recommendations..."):
    try:
        # Use hybrid mode if enough ratings, otherwise quick mode
        if len(ratings) >= 3:
            recommendations = st.session_state.recommender.get_hybrid_recommendations(
                ratings,
                preferred_genres=genres,
                time_preference=time_preference,
                discovery_type=discovery_type,
                n_recommendations=20
            )
        else:
            recommendations = st.session_state.recommender.get_quick_recommendations(
                preferred_genres=genres,
                time_preference=time_preference,
                discovery_type=discovery_type,
                n_recommendations=20
            )
        
        # Save recommendations in session
        st.session_state.recommendations = recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        raise e

# Display preference summary
with st.expander("📊 Your Preference Summary", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎬 Rated Movies")
        for movie_id, rating in ratings.items():
            title = movie_id_to_title.get(int(movie_id), str(movie_id))
            st.write(f"- {'⭐' * rating} {title}")
    
    with col2:
        st.subheader("🎭 Preferred Genres")
        for genre in genres:
            st.write(f"- {genre}")
    
    with col3:
        st.subheader("⚙️ Other Preferences")
        st.write(f"- **Period:** {time_preference}")
        st.write(f"- **Style:** {discovery_type}")

st.markdown("---")

# Recommendations section
st.header("✨ Movies Recommended for You")

# Check if we have recommendations
if not recommendations:
    st.warning("No recommendations were generated. Please try again.")
    st.stop()

# Create DataFrame for recommendations
rec_df = pd.DataFrame(recommendations)

# Sort by recommendation score
rec_df = rec_df.sort_values('hybrid_score', ascending=False)

# Display recommendations in a grid (3 movies per row)
for i in range(0, len(rec_df.head(9)), 3):  # Changed to show 9 movies (3 rows of 3)
    # Create three columns for each row
    col1, col2, col3 = st.columns(3)
    
    # First movie in the row
    with col1:
        if i < len(rec_df):
            rec = rec_df.iloc[i].to_dict()
            st.markdown("---")
            # Poster
            st.image(
                get_movie_poster_url(rec['title'], rec.get('year')),
                caption=f"Score: {rec['hybrid_score']*5:.1f}/5",
                width=200
            )
            # Movie details
            st.subheader(f"{i+1}. {rec['title']}")
            st.write(f"**Genres:** {format_genres(rec['genres'])}")
            if rec.get('year') != 'Unknown':
                st.write(f"**Year:** {rec['year']}")
    
    # Second movie in the row
    with col2:
        if i+1 < len(rec_df):
            rec = rec_df.iloc[i+1].to_dict()
            st.markdown("---")
            # Poster
            st.image(
                get_movie_poster_url(rec['title'], rec.get('year')),
                caption=f"Score: {rec['hybrid_score']*5:.1f}/5",
                width=200
            )
            # Movie details
            st.subheader(f"{i+2}. {rec['title']}")
            st.write(f"**Genres:** {format_genres(rec['genres'])}")
            if rec.get('year') != 'Unknown':
                st.write(f"**Year:** {rec['year']}")
    
    # Third movie in the row
    with col3:
        if i+2 < len(rec_df):
            rec = rec_df.iloc[i+2].to_dict()
            st.markdown("---")
            # Poster
            st.image(
                get_movie_poster_url(rec['title'], rec.get('year')),
                caption=f"Score: {rec['hybrid_score']*5:.1f}/5",
                width=200
            )
            # Movie details
            st.subheader(f"{i+3}. {rec['title']}")
            st.write(f"**Genres:** {format_genres(rec['genres'])}")
            if rec.get('year') != 'Unknown':
                st.write(f"**Year:** {rec['year']}")

# Action buttons at the bottom
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🔄 Regenerate Recommendations"):
        st.query_params.update(page="results")
with col2:
    if st.button("📝 Modify Preferences"):
        st.query_params.update(page="questionnaire")
with col3:
    if st.button("🏠 Back to Home"):
        st.query_params.update(page="home")
