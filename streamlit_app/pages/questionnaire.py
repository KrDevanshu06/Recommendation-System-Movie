"""
Questionnaire page to initialize user preferences
"""
import streamlit as st
import pandas as pd
from config import QUESTIONNAIRE_MOVIES, GENRES, TIME_PERIODS
from src.utils import get_movie_poster_url, format_genres

# Page configuration
st.set_page_config(
    page_title="Questionnaire - MovieLens Recommender",
    page_icon="🎯",
    layout="wide"
)

# Title and introduction
st.title("🎯 Complete Questionnaire")
st.markdown("---")

# Introduction
st.write("""
## Let's discover your movie preferences!

To offer you the best recommendations, we need to learn more about your tastes.
This questionnaire consists of 4 main sections:
""")

# Display steps
steps = [
    "🎬 **Popular Movies**: Rate the movies you've seen",
    "🎭 **Preferred Genres**: Select your favorite genres",
    "📅 **Preferred Period**: Choose your favorite era",
    "🎯 **Recommendation Style**: Define your preferences"
]

for step in steps:
    st.write(f"- {step}")

st.markdown("---")

# Initialize ratings in session if not already done
if 'questionnaire_ratings' not in st.session_state:
    st.session_state.questionnaire_ratings = {}

# Section 1: Popular movies to rate
st.header("1️⃣ Popular Movies")
st.write("""
Rate the movies you've seen on a scale of 1 to 5 stars.
If you haven't seen the movie, you can leave the default rating (3 stars).
""")

# Display movies in a grid with cards
cols = st.columns(2)
for i, movie in enumerate(QUESTIONNAIRE_MOVIES):
    with cols[i % 2]:
        with st.container():
            st.markdown("---")
            st.subheader(movie['title'])
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(get_movie_poster_url(movie['title']), width=150)
            with col2:
                rating = st.slider(
                    "Your rating",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.questionnaire_ratings.get(movie['movie_id'], 3),
                    key=f"q_rating_{movie['movie_id']}"
                )
                st.session_state.questionnaire_ratings[movie['movie_id']] = rating
                st.write(f"{'⭐' * rating}")

st.markdown("---")

# Section 2: Preferred genres
st.header("2️⃣ Preferred Genres")
st.write("""
Select 2 to 3 genres that you particularly enjoy.
These choices will help us refine your recommendations.
""")

selected_genres = st.multiselect(
    "Genres",
    options=GENRES,
    max_selections=3,
    help="Select your preferred genres (2-3 maximum)"
)

st.markdown("---")

# Section 3: Time period
st.header("3️⃣ Preferred Period")
st.write("""
Which era interests you the most?
Choose the period that best matches your taste.
""")

time_preference = st.radio(
    "Select a period",
    options=list(TIME_PERIODS.keys()),
    horizontal=True
)

st.markdown("---")

# Section 4: Discovery type
st.header("4️⃣ Recommendation Style")
st.write("""
What attracts you most in a movie?
Choose the recommendation style that matches your expectations.
""")

discovery_type = st.radio(
    "Choose your style",
    options=[
        "highly_rated",  # Highly rated movies
        "hidden_gems",   # Rare discoveries
        "popular",       # Popular movies
        "balanced"       # Balanced mix
    ],
    format_func=lambda x: {
        "highly_rated": "🏆 Highly rated movies",
        "hidden_gems": "💎 Hidden gems",
        "popular": "🔥 Popular movies",
        "balanced": "⚖️ Balanced mix"
    }[x],
    horizontal=True
)

st.markdown("---")

# Section 5: Submit button
st.header("✨ Generate my recommendations")

# Button to generate recommendations
if st.button("✨ Generate my recommendations", use_container_width=True):
    if len(selected_genres) < 2:
        st.error("⚠️ Please select at least 2 preferred genres")
    else:
        # Save preferences
        st.session_state.user_preferences = {
            'ratings': st.session_state.questionnaire_ratings,
            'genres': selected_genres,
            'time_preference': time_preference,
            'discovery_type': discovery_type
        }
        
        # Redirect to results page
        st.query_params.update(page="results")
        st.rerun()
