"""
Système de Recommandation Hybride (Collaborative + Content-Based)
"""
import pandas as pd
import numpy as np
from config import ML_PARAMS
from src.collaborative_filter import CollaborativeFilter
from src.content_filter import ContentBasedFilter

class HybridRecommender:
    def __init__(self):
        self.cf_model = CollaborativeFilter()
        self.cb_model = ContentBasedFilter()
        self.is_trained = False
        
    def train(self):
        """Entraîne les deux modèles"""
        print("🚀 Entraînement du système hybride...")
        
        try:
            # Entraîner les modèles
            print("Training collaborative filter...")
            self.cf_model.train_all()
            print("Training content-based filter...")
            self.cb_model.train_all()
            
            self.is_trained = True
            print("✅ Système hybride prêt!")
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement: {str(e)}")
            raise e
        
    def get_hybrid_recommendations(self, user_ratings, preferred_genres=None, 
                                 time_preference=None, discovery_type="balanced",
                                 n_recommendations=10):
        """
        Génère des recommandations hybrides
        
        Args:
            user_ratings: dict {movie_id: rating}
            preferred_genres: list des genres préférés
            time_preference: période temporelle préférée
            discovery_type: type de découverte ("highly_rated", "hidden_gems", "popular", "balanced")
            n_recommendations: nombre de recommandations
        """
        if not self.is_trained:
            self.train()
            
        # Ajouter l'utilisateur au système collaboratif
        new_user_id = self.cf_model.add_new_user(user_ratings)
        
        # Obtenir les recommandations de chaque système
        cf_recommendations = self.cf_model.get_user_recommendations(
            new_user_id, n_recommendations=n_recommendations*2, method='svd'
        )
        
        cb_recommendations = self.cb_model.get_content_recommendations(
            user_ratings, preferred_genres, time_preference, n_recommendations=n_recommendations*2
        )
        
        # Combiner les recommandations
        hybrid_recommendations = self._combine_recommendations(
            cf_recommendations, cb_recommendations, discovery_type
        )
        
        # Appliquer les filtres et ajustements
        final_recommendations = self._apply_filters_and_adjustments(
            hybrid_recommendations, preferred_genres, time_preference, discovery_type
        )
        
        return final_recommendations[:n_recommendations]
        
    def _combine_recommendations(self, cf_recs, cb_recs, discovery_type):
        """Combine les recommandations des deux systèmes"""
        # Créer un dictionnaire pour faciliter la fusion
        combined_recs = {}
        
        # Poids pour chaque système
        cf_weight = ML_PARAMS['cf_weight']  # 0.7
        cb_weight = ML_PARAMS['content_weight']  # 0.3
        
        # Ajouter les recommandations CF
        for i, rec in enumerate(cf_recs):
            movie_id = rec['movie_id']
            
            # Score basé sur la position + rating prédit
            position_score = (len(cf_recs) - i) / len(cf_recs)  # Score de position (1 à 0)
            cf_score = rec['predicted_rating'] / 5.0  # Normaliser entre 0 et 1
            final_cf_score = (position_score + cf_score) / 2
            
            combined_recs[movie_id] = {
                'movie_id': movie_id,
                'title': rec['title'],
                'genres': rec['genres'],
                'year': rec['year'],
                'cf_score': final_cf_score,
                'cb_score': 0,  # À remplir
                'hybrid_score': 0  # À calculer
            }
            
        # Ajouter les recommandations Content-Based
        for i, rec in enumerate(cb_recs):
            movie_id = rec['movie_id']
            
            # Score basé sur la position + score de contenu
            position_score = (len(cb_recs) - i) / len(cb_recs)
            cb_score = rec['content_score']
            final_cb_score = (position_score + cb_score) / 2
            
            if movie_id in combined_recs:
                combined_recs[movie_id]['cb_score'] = final_cb_score
            else:
                combined_recs[movie_id] = {
                    'movie_id': movie_id,
                    'title': rec['title'],
                    'genres': rec['genres'],
                    'year': rec['year'],
                    'cf_score': 0,
                    'cb_score': final_cb_score,
                    'hybrid_score': 0
                }
                
        # Calculer le score hybride final
        for movie_id in combined_recs:
            rec = combined_recs[movie_id]
            rec['hybrid_score'] = (cf_weight * rec['cf_score'] + 
                                 cb_weight * rec['cb_score'])
            
        # Convertir en liste et trier
        recommendations = list(combined_recs.values())
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return recommendations
        
    def _apply_filters_and_adjustments(self, recommendations, preferred_genres, 
                                     time_preference, discovery_type):
        """Applique les filtres et ajustements selon les préférences"""
        
        # Ajustements selon le type de découverte
        for rec in recommendations:
            if discovery_type == "highly_rated":
                # Bonus pour les films probablement bien notés
                if rec['cf_score'] > 0.8:
                    rec['hybrid_score'] += 0.1
                    
            elif discovery_type == "hidden_gems":
                # Bonus pour les films avec bon score content mais faible CF
                if rec['cb_score'] > 0.6 and rec['cf_score'] < 0.4:
                    rec['hybrid_score'] += 0.15
                    
            elif discovery_type == "popular":
                # Bonus pour les films avec fort score CF (populaires)
                if rec['cf_score'] > 0.6:
                    rec['hybrid_score'] += 0.1
                    
        # Bonus pour les genres préférés (déjà appliqué en partie dans CB)
        if preferred_genres:
            for rec in recommendations:
                if isinstance(rec['genres'], list):
                    common_genres = len(set(rec['genres']) & set(preferred_genres))
                    if common_genres > 0:
                        rec['hybrid_score'] += common_genres * 0.05
                        
        # Diversité des genres (éviter trop de films du même genre)
        recommendations = self._ensure_genre_diversity(recommendations)
        
        # Re-trier après ajustements
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return recommendations
        
    def _ensure_genre_diversity(self, recommendations, max_per_genre=3):
        """Assure la diversité des genres dans les recommandations"""
        genre_counts = {}
        diversified_recs = []
        
        for rec in recommendations:
            # Compter les genres principaux
            movie_genres = rec['genres'] if isinstance(rec['genres'], list) else []
            
            # Vérifier si on peut ajouter ce film sans dépasser la limite par genre
            can_add = True
            for genre in movie_genres:
                if genre in genre_counts and genre_counts[genre] >= max_per_genre:
                    can_add = False
                    break
                    
            if can_add:
                diversified_recs.append(rec)
                # Mettre à jour les compteurs
                for genre in movie_genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
                    
        return diversified_recs
        
    def get_explanation(self, recommendation):
        """Génère une explication pour une recommandation"""
        explanations = []
        
        # Explication basée sur les scores
        if recommendation['cf_score'] > 0.6:
            explanations.append("Aimé par des utilisateurs aux goûts similaires")
            
        if recommendation['cb_score'] > 0.6:
            explanations.append("Correspond à vos préférences de genres")
            
        # Explication basée sur les genres
        if isinstance(recommendation['genres'], list) and recommendation['genres']:
            genre_text = ", ".join(recommendation['genres'][:3])
            explanations.append(f"Genres: {genre_text}")
            
        # Explication basée sur l'année
        if recommendation['year'] != 'Unknown':
            explanations.append(f"Film de {recommendation['year']}")
            
        return " • ".join(explanations) if explanations else "Recommandation basée sur votre profil"
        
    def get_quick_recommendations(self, preferred_genres, time_preference=None, 
                                discovery_type="balanced", n_recommendations=10):
        """
        Recommandations rapides basées uniquement sur les préférences 
        (pour les utilisateurs sans ratings)
        """
        if not self.is_trained:
            self.train()
            
        # Utiliser principalement le content-based
        recommendations = self.cb_model.get_genre_recommendations(
            preferred_genres, n_recommendations=n_recommendations*2
        )
        
        # Ajouter quelques films populaires
        popular_movies = self.cf_model.get_popular_movies(n_recommendations//2)
        
        # Combiner et diversifier
        all_recs = []
        
        # Convertir les recommandations CB au format hybrid
        for rec in recommendations:
            all_recs.append({
                'movie_id': rec['movie_id'],
                'title': rec['title'],
                'genres': rec['genres'],
                'year': rec['year'],
                'cf_score': 0,
                'cb_score': rec['content_score'],
                'hybrid_score': rec['content_score']
            })
            
        # Ajouter les films populaires avec un score réduit
        for rec in popular_movies:
            if rec['movie_id'] not in [r['movie_id'] for r in all_recs]:
                all_recs.append({
                    'movie_id': rec['movie_id'],
                    'title': rec['title'],
                    'genres': rec['genres'],
                    'year': rec['year'],
                    'cf_score': rec['predicted_rating'] / 5.0,
                    'cb_score': 0,
                    'hybrid_score': (rec['predicted_rating'] / 5.0) * 0.6  # Score réduit
                })
                
        # Appliquer les filtres
        final_recs = self._apply_filters_and_adjustments(
            all_recs, preferred_genres, time_preference, discovery_type
        )
        
        return final_recs[:n_recommendations]

if __name__ == "__main__":
    # Test du système hybride
    hybrid = HybridRecommender()
    
    # Exemple de ratings utilisateur
    test_ratings = {
        1: 5,    # Toy Story
        50: 4,   # Star Wars  
        269: 3   # Fargo
    }
    
    recommendations = hybrid.get_hybrid_recommendations(
        test_ratings, 
        preferred_genres=['Action', 'Sci-Fi'],
        discovery_type="balanced"
    )
    
    print("🎬 RECOMMANDATIONS HYBRIDES:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec['title']} (Score: {rec['hybrid_score']:.3f})")
        print(f"   {hybrid.get_explanation(rec)}")
        print()