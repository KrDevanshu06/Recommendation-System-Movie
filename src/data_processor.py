"""
Preprocessing des données MovieLens
"""
import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MOVIELENS_URL

class MovieLensProcessor:
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.users = None
        
    def download_movielens(self):
        """Télécharge et extrait le dataset MovieLens 100k"""
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        
        zip_path = os.path.join(RAW_DATA_DIR, 'ml-100k.zip')
        
        if not os.path.exists(zip_path):
            print("📥 Téléchargement du dataset MovieLens...")
            urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
            
        # Extraction
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
            
        print("✅ Dataset téléchargé et extrait!")
        
    def load_data(self):
        """Charge les données MovieLens depuis les fichiers"""
        data_path = os.path.join(RAW_DATA_DIR, 'ml-100k')
        
        # Ratings (u.data)
        self.ratings = pd.read_csv(
            os.path.join(data_path, 'u.data'), 
            sep='\t', 
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'movie_id': int, 'user_id': int, 'rating': float}
        )        # Movies (u.item)
        column_names = ['movie_id', 'title', 'release_date', 'video_release_date', 'url']
        genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        self.movies = pd.read_csv(
            os.path.join(data_path, 'u.item'),
            sep='|', 
            encoding='latin-1',
            names=column_names + genre_cols,
            index_col=False
        )
        # Ensure movie_id is properly handled
        self.movies['movie_id'] = self.movies['movie_id'].astype(int)
        
        # Users (u.user)
        self.users = pd.read_csv(
            os.path.join(data_path, 'u.user'),
            sep='|', 
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            encoding='latin-1',
            dtype={'user_id': int}
        )
        
        print(f"✅ Données chargées:")
        print(f"   - {len(self.ratings)} ratings")
        print(f"   - {len(self.movies)} films")
        print(f"   - {len(self.users)} utilisateurs")
        
    def clean_data(self):
        """Nettoie et préprocess les données"""
        # Extraction de l'année de sortie
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')
        
        # Création de la matrice genres
        genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                      'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Création de la liste des genres pour chaque film
        genre_data = self.movies[genre_cols]
        self.movies['genres'] = genre_data.apply(
            lambda x: [col for col, val in zip(genre_cols, x) if val == 1],
            axis=1
        )
        
        # Nettoyage des ratings
        self.ratings = self.ratings.dropna()
        
        # Assurer que les movie_ids sont des entiers
        self.ratings['movie_id'] = self.ratings['movie_id'].astype(int)
        self.movies['movie_id'] = self.movies['movie_id'].astype(int)
        
        # Reset indices pour le merging
        self.ratings.reset_index(drop=True, inplace=True)
        self.movies.reset_index(drop=True, inplace=True)
        
        # Merge movies data into ratings
        self.ratings = self.ratings.merge(
            self.movies[['movie_id', 'title', 'year', 'genres']], 
            on='movie_id',
            how='inner'
        )
        
        print("✅ Données nettoyées!")
        
    def create_user_item_matrix(self):
        """Crée la matrice user-item pour le collaborative filtering"""
        user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        return user_item_matrix
        
    def get_movie_features(self):
        """Crée la matrice de features des films pour content-based"""
        genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                      'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        movie_features = self.movies[['movie_id', 'title', 'year'] + genre_cols].copy()
        
        # Normalisation de l'année
        movie_features['year_norm'] = (movie_features['year'] - movie_features['year'].min()) / \
                                      (movie_features['year'].max() - movie_features['year'].min())
        movie_features['year_norm'] = movie_features['year_norm'].fillna(0.5)
        
        return movie_features
        
    def save_processed_data(self):
        """Sauvegarde les données preprocessées"""
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        self.ratings.to_csv(os.path.join(PROCESSED_DATA_DIR, 'ratings_clean.csv'), index=False)
        self.movies.to_csv(os.path.join(PROCESSED_DATA_DIR, 'movies_clean.csv'), index=False)
        self.users.to_csv(os.path.join(PROCESSED_DATA_DIR, 'users_clean.csv'), index=False)
        
        # Matrice user-item
        user_item_matrix = self.create_user_item_matrix()
        user_item_matrix.to_csv(os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv'))
        
        # Features des films
        movie_features = self.get_movie_features()
        movie_features.to_csv(os.path.join(PROCESSED_DATA_DIR, 'movie_features.csv'), index=False)
        
        print("✅ Données sauvegardées dans", PROCESSED_DATA_DIR)
        
    def get_basic_stats(self):
        """Retourne des statistiques de base"""
        stats = {
            'n_users': self.ratings['user_id'].nunique(),
            'n_movies': self.ratings['movie_id'].nunique(), 
            'n_ratings': len(self.ratings),
            'sparsity': 1 - len(self.ratings) / (self.ratings['user_id'].nunique() * self.ratings['movie_id'].nunique()),
            'avg_rating': self.ratings['rating'].mean(),
            'rating_distribution': self.ratings['rating'].value_counts().sort_index()
        }
        return stats
        
    def process_all(self):
        """Pipeline complet de preprocessing"""
        print("🚀 Démarrage du preprocessing MovieLens...")
        
        self.download_movielens()
        self.load_data()
        self.clean_data()
        self.save_processed_data()
        
        stats = self.get_basic_stats()
        
        print("\n📊 STATISTIQUES FINALES:")
        print(f"   - Utilisateurs: {stats['n_users']}")
        print(f"   - Films: {stats['n_movies']}")  
        print(f"   - Ratings: {stats['n_ratings']}")
        print(f"   - Sparsité: {stats['sparsity']:.3f}")
        print(f"   - Rating moyen: {stats['avg_rating']:.2f}")
        
        return stats

if __name__ == "__main__":
    processor = MovieLensProcessor()
    processor.process_all()