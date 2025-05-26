"""
Fonctions utilitaires pour le système de recommandation
"""
import pandas as pd
import numpy as np
import requests
from PIL import Image
import os
from config import QUESTIONNAIRE_MOVIES, PROCESSED_DATA_DIR

# Remplacez 'YOUR_TMDB_API_KEY' par votre vraie clé API TMDb
TMDB_API_KEY = '367edee46db1f504be11a41d14e071d4'

def get_movie_poster_url(movie_title, year=None):
    """
    Récupère l'URL du poster d'un film via l'API TMDb.
    Si aucun poster n'est trouvé, retourne une image par défaut.
    """
    # Si le titre est vide, retourner l'image par défaut
    if not movie_title:
        return "https://via.placeholder.com/300x450/808080/FFFFFF?text=No+Poster"

    # Mapping des titres de films vers leurs IDs IMDB (pour les films du questionnaire)
    movie_to_imdb = {
        "Toy Story (1995)": "tt0114709",
        "Star Wars (1977)": "tt0076759",
        "Fargo (1996)": "tt0116282",
        "Contact (1997)": "tt0118884",
        "Return of the Jedi (1983)": "tt0086190",
        "Liar Liar (1997)": "tt0119528",
        "English Patient, The (1996)": "tt0116209",
        "Scream (1996)": "tt0117571",
        "Air Force One (1997)": "tt0118571",
        "Independence Day (1996)": "tt0116629"
    }

    try:
        # D'abord essayer avec l'ID IMDB si disponible
        imdb_id = movie_to_imdb.get(movie_title)
        if imdb_id:
            search_url = f"https://api.themoviedb.org/3/find/{imdb_id}"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "external_source": "imdb_id"
            }
        else:
            # Sinon, utiliser la recherche par titre et année
            search_url = "https://api.themoviedb.org/3/search/movie"
            # Nettoyer le titre (enlever l'année entre parenthèses)
            clean_title = movie_title.split(' (')[0].strip()
            params = {
                "api_key": TMDB_API_KEY,
                "query": clean_title,
                "language": "en-US",
                "include_adult": False
            }
            if year and year != 'Unknown':
                params["year"] = year

        response = requests.get(search_url, params=params)
        data = response.json()
        
        if imdb_id:
            results = data.get("movie_results", [])
        else:
            results = data.get("results", [])

        if results:
            movie_id = results[0]["id"]
            # Récupérer les détails du film pour obtenir le poster
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            details_params = {"api_key": TMDB_API_KEY}
            details_response = requests.get(details_url, params=details_params)
            details_data = details_response.json()
            poster_path = details_data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            else:
                print(f"Aucun poster trouvé pour {movie_title} (ID: {movie_id})")
        else:
            print(f"Aucun résultat trouvé pour {movie_title}")
    except Exception as e:
        print(f"Erreur lors de la récupération du poster pour {movie_title}: {e}")

    # Si aucun poster n'est trouvé, retourner l'image par défaut
    return "https://via.placeholder.com/300x450/808080/FFFFFF?text=No+Poster"

def format_genres(genres_list):
    """Formate la liste des genres pour l'affichage"""
    if isinstance(genres_list, str):
        try:
            genres_list = eval(genres_list)
        except:
            return genres_list
            
    if isinstance(genres_list, list):
        return " • ".join(genres_list[:3])  # Limite à 3 genres
    
    return str(genres_list)

def get_questionnaire_movies_info():
    """Récupère les informations des films du questionnaire"""
    try:
        movies_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'movies_clean.csv'))
        
        questionnaire_info = []
        for movie in QUESTIONNAIRE_MOVIES:
            movie_info = movies_df[movies_df['movie_id'] == movie['movie_id']]
            if not movie_info.empty:
                movie_data = movie_info.iloc[0]
                questionnaire_info.append({
                    'movie_id': movie['movie_id'],
                    'title': movie['title'],
                    'genres': format_genres(movie_data.get('genres', [])),
                    'year': int(movie_data['year']) if pd.notna(movie_data.get('year')) else 'Unknown',
                    'poster_url': get_movie_poster_url(movie['title'])
                })
                
        return questionnaire_info
        
    except Exception as e:
        print(f"Erreur lors du chargement des films du questionnaire: {e}")
        # Fallback avec données minimales
        return [
            {
                'movie_id': movie['movie_id'],
                'title': movie['title'], 
                'genres': 'Unknown',
                'year': 'Unknown',
                'poster_url': get_movie_poster_url(movie['title'])
            }
            for movie in QUESTIONNAIRE_MOVIES
        ]

def calculate_user_preferences_summary(user_ratings, preferred_genres, time_preference):
    """Calcule un résumé des préférences utilisateur"""
    summary = {
        'total_ratings': len(user_ratings),
        'avg_rating': np.mean(list(user_ratings.values())) if user_ratings else 0,
        'preferred_genres': preferred_genres or [],
        'time_preference': time_preference or 'Toutes époques',
        'rating_distribution': {}
    }
    
    # Distribution des ratings
    if user_ratings:
        for rating in user_ratings.values():
            summary['rating_distribution'][rating] = summary['rating_distribution'].get(rating, 0) + 1
            
    return summary

def format_recommendation_card(recommendation, include_explanation=True):
    """Formate une recommandation pour l'affichage"""
    card = {
        'title': recommendation['title'],
        'year': recommendation.get('year', 'Unknown'),
        'genres': format_genres(recommendation.get('genres', [])),
        'score': round(recommendation.get('hybrid_score', 0) * 5, 1),  # Convertir en note sur 5
        'poster_url': get_movie_poster_url(recommendation['title'], recommendation.get('year')),
        'movie_id': recommendation['movie_id']
    }
    
    return card

def get_recommendation_explanation(recommendation):
    """Génère une explication détaillée pour une recommandation"""
    explanations = []
    
    # Score collaboratif
    cf_score = recommendation.get('cf_score', 0)
    if cf_score > 0.7:
        explanations.append("🎯 Fortement recommandé par des utilisateurs aux goûts similaires")
    elif cf_score > 0.4:
        explanations.append("👥 Apprécié par des utilisateurs comme vous")
        
    # Score content-based
    cb_score = recommendation.get('cb_score', 0)
    if cb_score > 0.6:
        explanations.append("🎭 Correspond parfaitement à vos genres préférés")
    elif cb_score > 0.3:
        explanations.append("🎬 En lien avec vos préférences de style")
        
    # Genres
    genres = recommendation.get('genres', [])
    if isinstance(genres, list) and genres:
        main_genres = genres[:2]
        explanations.append(f"📝 Genres: {', '.join(main_genres)}")
        
    # Année
    year = recommendation.get('year')
    if year and year != 'Unknown':
        explanations.append(f"📅 Film de {year}")
        
    return explanations

def validate_user_input(user_ratings, preferred_genres, time_preference):
    """Valide les entrées utilisateur"""
    errors = []
    warnings = []
    
    # Validation des ratings
    if not user_ratings:
        errors.append("Veuillez noter au moins un film")
    elif len(user_ratings) < 3:
        warnings.append("Plus de films notés amélioreront vos recommandations")
        
    # Validation des genres
    if not preferred_genres:
        warnings.append("Sélectionner des genres préférés améliorera vos recommandations")
    
    # Validation de la période
    if not time_preference:
        warnings.append("Sélectionner une période temporelle affinera vos recommandations")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }