"""
Collaborative Filtering pour MovieLens
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from config import PROCESSED_DATA_DIR, ML_PARAMS

class CollaborativeFilter:
    def __init__(self):
        self.user_item_matrix = None
        self.svd_model = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_mean_ratings = None
        self.movies_df = None
        
    def load_data(self):
        """Charge les données preprocessées"""
        self.user_item_matrix = pd.read_csv(
            os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv'),
            index_col=0
        )
        
        self.movies_df = pd.read_csv(
            os.path.join(PROCESSED_DATA_DIR, 'movies_clean.csv')
        )
        
        # Moyennes par utilisateur pour centrer les données
        self.user_mean_ratings = self.user_item_matrix.mean(axis=1)
        
        print(f"✅ Matrice user-item chargée: {self.user_item_matrix.shape}")
        
    def train_svd(self, n_components=50):
        """Entraîne le modèle SVD pour matrix factorization"""
        # Centrer les données (soustraire la moyenne utilisateur)
        user_item_centered = self.user_item_matrix.sub(self.user_mean_ratings, axis=0)
        user_item_centered = user_item_centered.fillna(0)
        
        # SVD
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = self.svd_model.fit_transform(user_item_centered)
        
        print(f"✅ SVD entraîné avec {n_components} facteurs")
        print(f"   - Variance expliquée: {self.svd_model.explained_variance_ratio_.sum():.3f}")
        
        return user_factors
        
    def compute_user_similarity(self):
        """Calcule la similarité entre utilisateurs"""
        # Remplacer les 0 par NaN pour ignorer les films non vus
        matrix_for_similarity = self.user_item_matrix.replace(0, np.nan)
        
        # Calculer la similarité cosinus
        self.user_similarity = pd.DataFrame(
            cosine_similarity(matrix_for_similarity.fillna(0)),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print("✅ Similarité utilisateurs calculée")
        
    def compute_item_similarity(self):
        """Calcule la similarité entre films"""
        matrix_for_similarity = self.user_item_matrix.T.replace(0, np.nan)
        
        self.item_similarity = pd.DataFrame(
            cosine_similarity(matrix_for_similarity.fillna(0)),
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        print("✅ Similarité items calculée")
        
    def predict_rating_user_based(self, user_id, movie_id, k=50):
        """Prédiction basée sur la similarité utilisateur"""
        if user_id not in self.user_similarity.index:
            return self.user_mean_ratings.mean()
            
        if movie_id not in self.user_item_matrix.columns:
            return self.user_mean_ratings.mean()
            
        # Utilisateurs similaires qui ont vu ce film
        movie_ratings = self.user_item_matrix[movie_id]
        movie_ratings = movie_ratings[movie_ratings > 0]
        
        if len(movie_ratings) == 0:
            return self.user_mean_ratings[user_id]
            
        # Top-k utilisateurs similaires
        user_similarities = self.user_similarity.loc[user_id]
        similar_users = user_similarities[movie_ratings.index].nlargest(k)
        
        if similar_users.sum() == 0:
            return self.user_mean_ratings[user_id]
            
        # Prédiction pondérée
        weighted_ratings = (similar_users * movie_ratings[similar_users.index]).sum()
        prediction = weighted_ratings / similar_users.sum()
        
        return prediction
        
    def predict_rating_svd(self, user_id, movie_id):
        """Prédiction avec SVD"""
        if user_id not in self.user_item_matrix.index:
            return self.user_mean_ratings.mean()
            
        if movie_id not in self.user_item_matrix.columns:
            return self.user_mean_ratings.mean()
            
        # Reconstruction via SVD
        user_idx = list(self.user_item_matrix.index).index(user_id)
        movie_idx = list(self.user_item_matrix.columns).index(movie_id)
        
        user_item_centered = self.user_item_matrix.sub(self.user_mean_ratings, axis=0)
        user_item_centered = user_item_centered.fillna(0)
        
        # Transform et inverse transform pour reconstruction
        user_factors = self.svd_model.transform(user_item_centered)
        reconstructed = self.svd_model.inverse_transform(user_factors)
        
        # Ajouter la moyenne utilisateur
        prediction = reconstructed[user_idx, movie_idx] + self.user_mean_ratings[user_id]
        
        # Borner entre 1 et 5
        prediction = max(1, min(5, prediction))
        
        return prediction
        
    def get_user_recommendations(self, user_id, n_recommendations=10, method='svd'):
        """Génère des recommandations pour un utilisateur"""
        if user_id not in self.user_item_matrix.index:
            return self.get_popular_movies(n_recommendations)
            
        # Films déjà vus par l'utilisateur
        seen_movies = self.user_item_matrix.loc[user_id]
        seen_movies = seen_movies[seen_movies > 0].index.tolist()
        
        # Films candidats (non vus)
        all_movies = self.user_item_matrix.columns.tolist()
        candidate_movies = [m for m in all_movies if m not in seen_movies]
        
        # Prédictions pour tous les films candidats
        predictions = []
        for movie_id in candidate_movies:
            if method == 'svd':
                pred = self.predict_rating_svd(user_id, movie_id)
            else:
                pred = self.predict_rating_user_based(user_id, movie_id)
                
            predictions.append((movie_id, pred))
            
        # Trier par prédiction décroissante
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Top N recommandations avec infos films
        recommendations = []
        for movie_id, pred_rating in predictions[:n_recommendations]:
            try:
                movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
                if not movie_info.empty:
                    movie_info = movie_info.iloc[0]
                    recommendations.append({
                        'movie_id': movie_id,
                        'title': movie_info['title'],
                        'predicted_rating': pred_rating,
                        'genres': movie_info['genres'] if 'genres' in movie_info else [],
                        'year': movie_info['year'] if pd.notna(movie_info['year']) else 'Unknown'
                    })
            except Exception as e:
                print(f"Warning: Could not find information for movie {movie_id}")
                continue
            
        return recommendations
        
    def add_new_user(self, user_ratings):
        """Ajoute un nouvel utilisateur avec ses ratings"""
        # Nouvel ID utilisateur
        new_user_id = self.user_item_matrix.index.max() + 1
        
        # Créer le vecteur de ratings pour ce nouvel utilisateur
        new_user_vector = pd.Series(0, index=self.user_item_matrix.columns)
        
        for movie_id, rating in user_ratings.items():
            if movie_id in new_user_vector.index:
                new_user_vector[movie_id] = rating
                
        # Ajouter à la matrice
        new_row = pd.DataFrame([new_user_vector], index=[new_user_id])
        self.user_item_matrix = pd.concat([self.user_item_matrix, new_row])
        
        # Recalculer la moyenne pour ce nouvel utilisateur
        self.user_mean_ratings[new_user_id] = new_user_vector[new_user_vector > 0].mean()
        
        print(f"✅ Nouvel utilisateur {new_user_id} ajouté avec {len(user_ratings)} ratings")
        
        return new_user_id
        
    def get_popular_movies(self, n=10):
        """Retourne les films les plus populaires (fallback)"""
        ratings_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'ratings_clean.csv'))
        
        popular = ratings_df.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).round(2)
        
        popular.columns = ['avg_rating', 'n_ratings']
        popular = popular[popular['n_ratings'] >= 50]  # Minimum 50 ratings
        popular = popular.sort_values(['avg_rating', 'n_ratings'], ascending=False)
        
        recommendations = []
        for movie_id in popular.head(n).index:
            try:
                movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
                if not movie_info.empty:
                    movie_info = movie_info.iloc[0]
                    recommendations.append({
                        'movie_id': movie_id,
                        'title': movie_info['title'],
                        'predicted_rating': popular.loc[movie_id, 'avg_rating'],
                        'genres': movie_info['genres'] if 'genres' in movie_info else [],
                        'year': movie_info['year'] if pd.notna(movie_info['year']) else 'Unknown'
                    })
            except Exception as e:
                print(f"Warning: Could not find information for movie {movie_id}")
                continue
            
        return recommendations
        
    def evaluate_model(self, test_size=0.2):
        """Évalue le modèle CF sur un test set"""
        ratings_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'ratings_clean.csv'))
        
        train_data, test_data = train_test_split(ratings_df, test_size=test_size, random_state=42)
        
        predictions = []
        actuals = []
        
        print("🔄 Évaluation du modèle...")
        
        for _, row in test_data.head(1000).iterrows():  # Échantillon pour rapidité
            user_id = row['user_id']
            movie_id = row['movie_id']  
            actual_rating = row['rating']
            
            pred_rating = self.predict_rating_svd(user_id, movie_id)
            
            predictions.append(pred_rating)
            actuals.append(actual_rating)
            
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
        
        print(f"📊 ÉVALUATION:")
        print(f"   - RMSE: {rmse:.3f}")
        print(f"   - MAE: {mae:.3f}")
        
        return {'rmse': rmse, 'mae': mae}
        
    def train_all(self):
        """Pipeline complet d'entraînement"""
        print("🚀 Entraînement Collaborative Filtering...")
        
        self.load_data()
        self.train_svd(n_components=ML_PARAMS['svd_factors'])
        self.compute_user_similarity()
        
        # Évaluation
        metrics = self.evaluate_model()
        
        print("✅ Collaborative Filtering prêt!")
        return metrics

if __name__ == "__main__":
    cf = CollaborativeFilter()
    cf.train_all()