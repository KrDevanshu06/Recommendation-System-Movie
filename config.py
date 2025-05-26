"""
Configuration globale du projet MovieLens Recommender
"""
import os

# Chemins des fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

__all__ = [
    'BASE_DIR', 'DATA_DIR', 'RAW_DATA_DIR', 
    'PROCESSED_DATA_DIR', 'MODELS_DIR', 'MOVIELENS_URL',
    'QUESTIONNAIRE_MOVIES', 'GENRES', 'TIME_PERIODS', 'ML_PARAMS',
    'STREAMLIT_CONFIG'
]

# URLs MovieLens Dataset
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Films populaires pour le questionnaire (garantis dans MovieLens 100k)
QUESTIONNAIRE_MOVIES = [
    {"title": "Toy Story (1995)", "movie_id": 1},
    {"title": "Star Wars (1977)", "movie_id": 50},
    {"title": "Fargo (1996)", "movie_id": 269},
    {"title": "Contact (1997)", "movie_id": 258},
    {"title": "Return of the Jedi (1983)", "movie_id": 181},
    {"title": "Liar Liar (1997)", "movie_id": 286},
    {"title": "English Patient, The (1996)", "movie_id": 288},
    {"title": "Scream (1996)", "movie_id": 300},
    {"title": "Air Force One (1997)", "movie_id": 313},
    {"title": "Independence Day (1996)", "movie_id": 324}
]

# Genres MovieLens
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",  
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
    "Thriller", "War", "Western"
]

# Périodes temporelles
TIME_PERIODS = {
    "Classiques": (0, 1980),
    "Golden Age": (1980, 1995), 
    "Modernes": (1995, 2000),
    "Toutes époques": (0, 2030)
}

# Paramètres ML
ML_PARAMS = {
    "n_recommendations": 10,
    "cf_weight": 0.7,
    "content_weight": 0.3,
    "min_ratings": 5,
    "svd_factors": 50
}

# Configuration Streamlit
STREAMLIT_CONFIG = {
    "page_title": "MovieLens Recommender 🎬",
    "page_icon": "🎬",
    "layout": "wide"
}