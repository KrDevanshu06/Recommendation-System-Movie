"""
Main Streamlit interface for the MovieLens recommendation system
"""
import streamlit as st
import pandas as pd
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path setup
from src.hybrid_recommender import HybridRecommender
from src.utils import (
    get_movie_poster_url, 
    format_genres,
    get_questionnaire_movies_info,
    calculate_user_preferences_summary
)
from config import GENRES, TIME_PERIODS, STREAMLIT_CONFIG

# Custom CSS for slider color
st.markdown("""
    <style>
    /* Slider bar background */
    .stSlider > div[data-baseweb="slider"] > div {
        background: #82B2C0 !important;
    }
    /* Slider active bar */
    .stSlider .css-1gv0vcd .css-14xtw13 {
        background: #4a7c8c !important;
    }
    /* Slider thumb */
    .stSlider .css-1gv0vcd .css-1eoe787 {
        background: #4a7c8c !important;
        border: 2px solid #23323a !important;
    }
    </style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"]
)

# Initialisation de la session
if 'recommender' not in st.session_state:
    st.session_state.recommender = HybridRecommender()
    st.session_state.recommender.train()

if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# Titre principal avec style
st.title("🎬 MovieLens Recommender")
st.markdown("---")

# En-tête avec image et description
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://raw.githubusercontent.com/grouplens/movielens/master/movielens-logo.png", width=200)
with col2:
    st.write("""
    ### Welcome to your personalized movie recommendation system!
    
    Our system uses a hybrid approach combining:
    - 🎯 Collaborative filtering
    - 🎭 Content-based filtering
    - 🤖 Machine learning algorithms
    
    To get started, fill out our questionnaire and discover movies tailored to your tastes!
    """)

st.markdown("---")

# Section principale
if st.session_state.recommendations:
    # Afficher les recommandations existantes
    st.header("🎬 Your Latest Recommendations")
    
    # Résumé des préférences
    with st.expander("📊 Your Preferences Summary", expanded=True):
        prefs_summary = calculate_user_preferences_summary(
            st.session_state.user_ratings,
            st.session_state.get('preferred_genres', []),
            st.session_state.get('time_preference', 'All periods')
        )
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rated movies", prefs_summary['total_ratings'])
        with col2:
            st.metric("Average rating", f"{prefs_summary['avg_rating']:.1f}/5")
        with col3:
            st.metric("Preferred genres", len(prefs_summary['preferred_genres']))
    
    # Afficher les recommandations
    for i, rec in enumerate(st.session_state.recommendations[:5], 1):
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(
                    get_movie_poster_url(rec['title'], rec.get('year')),
                    caption=f"Score: {rec['hybrid_score']*5:.1f}/5",
                    width=200
                )
                
            with col2:
                st.subheader(f"{i}. {rec['title']}")
                st.write(f"**Genres:** {format_genres(rec['genres'])}")
                if rec.get('year') != 'Unknown':
                    st.write(f"**Year:** {rec['year']}")
                
                # Explanation
                explanation = st.session_state.recommender.get_explanation(rec)
                st.info(" • ".join(explanation))
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Edit my preferences"):
            st.experimental_set_query_params(page="questionnaire")
    with col2:
        if st.button("🎬 See all my recommendations"):
            st.experimental_set_query_params(page="results")
else:
    # Message d'accueil pour les nouveaux utilisateurs
    st.header("👋 Start your movie journey!")
    
    # Présentation des fonctionnalités
    st.write("""
    ### How does it work?
    
    1. **📝 Personalized Questionnaire**
       - Rate movies you've seen
       - Select your favorite genres
       - Choose your favorite period
       - Define your discovery style
    
    2. **🎯 Smart Recommendations**
       - Movies tailored to your tastes
       - Surprising discoveries
       - Detailed explanations
       - Real-time updates
    
    3. **✨ Personalized Experience**
       - Intuitive interface
       - Evolving recommendations
       - History of your preferences
       - Always relevant suggestions
    """)
    
    # Bouton d'action principal
    st.markdown("---")
    if st.button("🚀 Start the questionnaire", use_container_width=True):
        st.experimental_set_query_params(page="questionnaire")

# Pied de page
st.markdown("---")
st.write("""
<div style='text-align: center'>
    <p>🎬 MovieLens Recommender - Your personal movie guide</p>
    <p>Based on the MovieLens 100k dataset</p>
</div>
""", unsafe_allow_html=True)
