"""
Content-Based Filtering pour MovieLens
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os
from config import PROCESSED_DATA_DIR, GENRES, TIME_PERIODS

class ContentBasedFilter:
    def __init__(self):
        self.movies_df = None
        self.movie_features = None
        self.content_similarity = None
        self.genre_vectorizer = None
        self.scaler = None
        
    def load_data(self):
        """Charge les données"""
        self.movies_df = pd.read_csv(
            os.path.join(PROCESSED_DATA_DIR, 'movies_clean.csv')
        )
        
        self.movie_features = pd.read_csv(
            os.path.join(PROCESSED_DATA_DIR, 'movie_features.csv')
        )
        
        print(f"✅ {len(self.movies_df)} films chargés pour content-based filtering")
        
    def prepare_features(self):
        """Prépare les features pour le content-based filtering"""
        # Features des genres (one-hot encoding déjà fait)
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                      'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Matrice des features finales
        features_matrix = []
        
        for _, movie in self.movie_features.iterrows():
            # Vecteur genre (18 dimensions)
            genre_vector = [movie[genre] for genre in genre_cols]
            
            # Feature année normalisée (1 dimension)
            year_feature = [movie['year_norm']]
            
            # Combiner toutes les features
            movie_vector = genre_vector + year_feature
            features_matrix.append(movie_vector)
            
        self.features_matrix = np.array(features_matrix)
        
        print(f"✅ Matrice de features créée: {self.features_matrix.shape}")
        
    def compute_content_similarity(self):
        """Calcule la similarité basée sur le contenu"""
        # Similarité cosinus entre tous les films
        self.content_similarity = cosine_similarity(self.features_matrix)
        
        # Convertir en DataFrame pour facilité d'usage
        self.content_similarity = pd.DataFrame(
            self.content_similarity,
            index=self.movie_features['movie_id'].values,
            columns=self.movie_features['movie_id'].values
        )
        
        print("✅ Similarité de contenu calculée")
        
    def get_similar_movies(self, movie_id, n_similar=10):
        """Trouve les films similaires à un film donné"""
        if movie_id not in self.content_similarity.index:
            return []
            
        # Films similaires (exclure le film lui-même)
        similar_scores = self.content_similarity.loc[movie_id].drop(movie_id)
        similar_movies = similar_scores.nlargest(n_similar)
        
        results = []
        for sim_movie_id, similarity in similar_movies.items():
            movie_info = self.movies_df[self.movies_df['movie_id'] == sim_movie_id]
            if not movie_info.empty:
                movie_info = movie_info.iloc[0]
                results.append({
                    'movie_id': sim_movie_id,
                    'title': movie_info['title'],
                    'similarity': similarity,
                    'genres': movie_info['genres'] if 'genres' in movie_info else [],
                    'year': movie_info['year'] if pd.notna(movie_info['year']) else 'Unknown'
                })
                
        return results
        
    def get_content_recommendations(self, user_ratings, preferred_genres=None, 
                                 time_preference=None, n_recommendations=10):
        """
        Génère des recommandations basées sur le contenu
        
        Args:
            user_ratings: dict {movie_id: rating}
            preferred_genres: list des genres préférés
            time_preference: période temporelle préférée
            n_recommendations: nombre de recommandations
        """
        # Films vus par l'utilisateur
        seen_movies = list(user_ratings.keys())
        
        # Calcul du profil utilisateur basé sur ses ratings
        user_profile = self._build_user_profile(user_ratings)
        
        # Films candidats (non vus)
        all_movies = self.movie_features['movie_id'].tolist()
        candidate_movies = [m for m in all_movies if m not in seen_movies]
        
        # Calcul des scores pour chaque film candidat
        recommendations = []
        
        for movie_id in candidate_movies:
            # Score basé sur la similarité avec le profil utilisateur
            content_score = self._calculate_content_score(movie_id, user_profile)
            
            # Bonus pour les genres préférés
            if preferred_genres:
                genre_bonus = self._calculate_genre_bonus(movie_id, preferred_genres)
                content_score += genre_bonus
                
            # Bonus pour la période temporelle
            if time_preference and time_preference != "Toutes époques":
                time_bonus = self._calculate_time_bonus(movie_id, time_preference)
                content_score += time_bonus
                
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if not movie_info.empty:
                movie_info = movie_info.iloc[0]
                
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'],
                    'content_score': content_score,
                    'genres': movie_info['genres'] if 'genres' in movie_info else [],
                    'year': movie_info['year'] if pd.notna(movie_info['year']) else 'Unknown'
                })
                
        # Trier par score décroissant
        recommendations.sort(key=lambda x: x['content_score'], reverse=True)
        
        return recommendations[:n_recommendations]
        
    def _build_user_profile(self, user_ratings):
        """Construit le profil utilisateur basé sur ses ratings"""
        profile = np.zeros(self.features_matrix.shape[1])
        total_weight = 0
        
        for movie_id, rating in user_ratings.items():
            movie_idx = self._get_movie_index(movie_id)
            if movie_idx is not None:
                # Pondérer par le rating (films mieux notés ont plus d'influence)
                weight = rating / 5.0  # Normaliser entre 0 et 1
                profile += weight * self.features_matrix[movie_idx]
                total_weight += weight
                
        if total_weight > 0:
            profile /= total_weight
            
        return profile
        
    def _calculate_content_score(self, movie_id, user_profile):
        """Calcule le score de similarité avec le profil utilisateur"""
        movie_idx = self._get_movie_index(movie_id)
        if movie_idx is None:
            return 0
            
        movie_vector = self.features_matrix[movie_idx]
        
        # Similarité cosinus
        similarity = np.dot(user_profile, movie_vector) / (
            np.linalg.norm(user_profile) * np.linalg.norm(movie_vector) + 1e-8
        )
        
        return similarity
        
    def _calculate_genre_bonus(self, movie_id, preferred_genres):
        """Calcule le bonus pour les genres préférés"""
        movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
        if movie_info.empty:
            return 0
            
        movie_genres = movie_info.iloc[0]['genres']
        if isinstance(movie_genres, str):
            # Si c'est une string, la convertir en liste
            movie_genres = eval(movie_genres) if movie_genres.startswith('[') else [movie_genres]
        elif not isinstance(movie_genres, list):
            return 0
            
        # Bonus proportionnel au nombre de genres en commun
        common_genres = len(set(movie_genres) & set(preferred_genres))
        bonus = common_genres * 0.2  # 0.2 par genre en commun
        
        return bonus
        
    def _calculate_time_bonus(self, movie_id, time_preference):
        """Calcule le bonus pour la période temporelle"""
        movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
        if movie_info.empty:
            return 0
            
        movie_year = movie_info.iloc[0]['year']
        if pd.isna(movie_year):
            return 0
            
        # Vérifier si le film est dans la période préférée
        if time_preference in TIME_PERIODS:
            min_year, max_year = TIME_PERIODS[time_preference]
            if min_year <= movie_year <= max_year:
                return 0.1  # Bonus de 0.1 pour la bonne période
                
        return 0
        
    def _get_movie_index(self, movie_id):
        """Récupère l'index d'un film dans la matrice de features"""
        try:
            return list(self.movie_features['movie_id']).index(movie_id)
        except ValueError:
            return None
            
    def get_genre_recommendations(self, preferred_genres, n_recommendations=10):
        """Recommandations basées uniquement sur les genres préférés"""
        recommendations = []
        
        for _, movie in self.movies_df.iterrows():
            movie_genres = movie['genres']
            if isinstance(movie_genres, str):
                movie_genres = eval(movie_genres) if movie_genres.startswith('[') else [movie_genres]
            elif not isinstance(movie_genres, list):
                continue
                
            # Score basé sur le nombre de genres en commun
            common_genres = len(set(movie_genres) & set(preferred_genres))
            if common_genres > 0:
                recommendations.append({
                    'movie_id': movie['movie_id'],
                    'title': movie['title'],
                    'content_score': common_genres / len(preferred_genres),
                    'genres': movie_genres,
                    'year': movie['year'] if pd.notna(movie['year']) else 'Unknown'
                })
                
        # Trier par score décroissant
        recommendations.sort(key=lambda x: x['content_score'], reverse=True)
        
        return recommendations[:n_recommendations]
        
    def train_all(self):
        """Pipeline complet d'entraînement"""
        print("🚀 Entraînement Content-Based Filter...")
        
        self.load_data()
        self.prepare_features()
        self.compute_content_similarity()
        
        print("✅ Content-Based Filter prêt!")
        
        # Test rapide
        sample_movie_id = self.movies_df.iloc[0]['movie_id']
        similar = self.get_similar_movies(sample_movie_id, 5)
        print(f"📝 Test: Films similaires à '{self.movies_df.iloc[0]['title']}': {len(similar)} trouvés")

if __name__ == "__main__":
    cbf = ContentBasedFilter()
    cbf.train_all()