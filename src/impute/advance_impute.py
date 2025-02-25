import numpy as np
import time
import tracemalloc

from src.utils.metrics_utils import score_imputed

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import ParameterGrid, KFold

import warnings
warnings.filterwarnings("ignore")

class BaseImputer:
    def __init__(self, column, type, **kwargs):
        self.column = column
        self.type = type
        self.kwargs = kwargs

    def fit_transform(self, X):
        raise NotImplementedError("Chaque classe doit implémenter fit_transform")

# Méthode KNN
class KNNImputerMethod(BaseImputer):
    def fit_transform(self, X):
        imputer = KNNImputer(**self.kwargs)
        return imputer.fit_transform(X)

# Méthode SofImputer
class SoftImputerMethod(BaseImputer):
    def fit_transform(self, X):
        return self.soft_imput_implementation(X, **self.kwargs)
    
    def soft_imput_implementation(self, X, lambda_val=1e-6, tol=1e-5, max_iter=500):
        Omega = ~np.isnan(X)
        X_filled = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        for _ in range(max_iter):
            U, singular_values, Vt = np.linalg.svd(X_filled, full_matrices=False)
            singular_values_tilde = np.maximum(0, singular_values - lambda_val)
            Sigma_tilde = np.diag(singular_values_tilde)
            X_hat = U @ Sigma_tilde @ Vt
            X_hat[Omega] = X[Omega]
            
            if np.linalg.norm(X_hat - X_filled, ord='fro') < tol:
                break
            
            X_filled = X_hat
        
        return X_hat
    
# Méthode ACP    
class ACPImputerMethod(BaseImputer):
    def fit_transform(self, X):
        return self.acp_impute(X, self.column, **self.kwargs)

    def acp_impute(self, data, idx_col, n_components=2, max_iter=500, min_mse=1e-4):
        n_components = min(n_components, min(data.shape) - 1)  # Limitation dynamique
        idx_NA = np.where(np.isnan(data[:, idx_col]))[0]
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)
        previous = data_imputed.copy()
        
        for _ in range(max_iter):
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(previous)
            data_reconstructed = pca.inverse_transform(principal_components)
            data_imputed[idx_NA, idx_col] = data_reconstructed[idx_NA, idx_col]
            
            diff = np.mean((data_imputed[idx_NA, idx_col] - previous[idx_NA, idx_col]) ** 2)
            if diff < min_mse:
                break
            previous = data_imputed.copy()
        
        return data_imputed

# Méthode ICE
class ICEImputerMethod(BaseImputer):
    def fit_transform(self, X):
        return self.ice_impute(X, self.column, self.type, **self.kwargs)
    
    def ice_impute(self, data, idx_col, type, max_iter=500, tol=1e-4):
        # Initialisation des valeurs manquantes
        initial_imputer = SimpleImputer(strategy='mean' if type == 'num' else 'most_frequent')
        data_imputed = initial_imputer.fit_transform(data)
        
        nan_mask = np.isnan(data[:, idx_col])
        X_train = np.delete(data_imputed, idx_col, axis=1)[~nan_mask]
        y_train = data_imputed[:, idx_col][~nan_mask]
        X_test = np.delete(data_imputed, idx_col, axis=1)[nan_mask]
        
        for _ in range(max_iter):
            # Entraînement du modèle
            if type == 'num':
                model = LinearRegression()
            else:
                model = LogisticRegression(max_iter=500)
            model.fit(X_train, y_train)
            
            # Prédiction des valeurs manquantes
            y_pred = model.predict(X_test)
            previous_values = data_imputed[nan_mask, idx_col].copy()
            data_imputed[nan_mask, idx_col] = y_pred
            
            # Vérification de la convergence
            if np.linalg.norm(data_imputed[nan_mask, idx_col] - previous_values) < tol:
                break
            
            # Mise à jour des données d'entraînement pour la prochaine itération
            X_train = np.delete(data_imputed, idx_col, axis=1)[~nan_mask]
            y_train = data_imputed[:, idx_col][~nan_mask]
        
        return data_imputed

# Méthode MissForest
class MissForestImputerMethod(BaseImputer):
    def fit_transform(self, X):
        return self.missforest_impute(X, self.column, self.type, **self.kwargs)
    
    def missforest_impute(self, X, idx_col, type, max_iter=500, n_estimators=10, max_depth=20, min_samples_split=5, min_samples_leaf=2):
        X_imputed = X.copy()
        nan_mask = np.isnan(X_imputed)
        missing_rows = nan_mask[:, idx_col]

        # Initialisation des valeurs manquantes avec SimpleImputer
        imputer = SimpleImputer(strategy="median" if type == "num" else "most_frequent")
        X_imputed[:, idx_col] = imputer.fit_transform(X_imputed[:, idx_col].reshape(-1, 1)).ravel()

        for iteration in range(max_iter):
            X_prev = X_imputed.copy()

            # Extraction des données sans valeurs manquantes
            X_train = np.delete(X_imputed[~missing_rows], idx_col, axis=1)
            y_train = X_imputed[~missing_rows, idx_col]
            X_pred = np.delete(X_imputed[missing_rows], idx_col, axis=1)

            # Choix du modèle
            if type == 'num':
                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    random_state=42
                )

            # Entraînement et prédiction
            model.fit(X_train, y_train)
            predictions = model.predict(X_pred)
            X_imputed[missing_rows, idx_col] = predictions

            # Vérification de la convergence avec norme de Frobenius
            if np.linalg.norm(X_imputed - X_prev, ord='fro') < 1e-5:
                break

        return X_imputed


def advance_imputation(data_original: np.ndarray, data_ampute: np.ndarray, column: int, idx_NA: list, type: str, method: str):
    """
    Effectue une imputation avancée des valeurs manquantes en utilisant différentes méthodes d'imputation et sélectionne les meilleurs hyperparamètres via validation croisée.

    Paramètres :
    ----------
    data_original : np.ndarray
        Données d'origine complètes (sans valeurs manquantes) pour l'évaluation.
    data_ampute : np.ndarray
        Données avec valeurs manquantes à imputer.
    column : int
        Indice de la colonne ciblée pour l'imputation.
    idx_NA : list
        Liste des indices des valeurs manquantes dans la colonne spécifiée.
    type : str
        Type de données à imputer ('num' pour numérique, 'cat' pour catégorielle).
    method : str
        Méthode d'imputation à utiliser ('KNN', 'SoftImputer', 'ICE', 'MissForest', 'ACP').

    Retourne :
    ---------
    best_score : float
        Meilleur score d'imputation obtenu lors de la validation croisée.
    exec_time : float
        Temps d'exécution total pour l'imputation (en secondes).
    mem_used : float
        Mémoire maximale utilisée pendant l'imputation (en Mo).
    best_imputed_data : np.ndarray
        Données imputées avec les meilleurs hyperparamètres.
    """

    imputers = {
        "KNN": KNNImputerMethod,
        "SoftImputer": SoftImputerMethod,
        "ACP": ACPImputerMethod,
        "ICE": ICEImputerMethod,
        "MissForest": MissForestImputerMethod
    }
    
    param_grids = {
        "KNN": {
            "n_neighbors": [1, 2, 3, 5, 10, 20, 50],
            "weights": ["uniform", "distance"]
        },
        "SoftImputer": {
            "lambda_val": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "tol": [1e-7, 1e-6, 1e-5]
        },
        "ACP": {
            "n_components": [1, 2, 3, 5, 10, 20, 50]
        },
        "ICE": {
            "tol": [1e-7, 1e-6, 1e-5]
        },
        "MissForest": {
            "n_estimators": [1, 2, 3, 5, 10, 20, 50]
        }
    }
    
    # Initialisation du meilleur score en fonction du type de variable :
    # - Pour les variables numériques ('num'), on cherche à minimiser la MSE, donc on initialise à +inf.
    # - Pour les variables catégorielles ('cat'), on cherche à maximiser l'accuracy, donc on initialise à -inf.
    best_score = float("inf") if type == "num" else float("-inf")
    best_params = None
    best_imputed_data = None
    
    # Validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Boucle sur toutes les combinaisons de paramètres possibles pour la méthode d'imputation sélectionnée
    for params in ParameterGrid(param_grids[method]):
        scores = [] # Score de chaque pli

        for train_idx, test_idx in kf.split(data_ampute):
            # Évaluation de l'imputation en comparant les valeurs imputées aux données originales pour ce pli
            imputer = imputers[method](column, type, **params)
            data_imputed = imputer.fit_transform(data_ampute.copy())
            idx_NA_test = [c for c in idx_NA if c in test_idx]
            score = score_imputed(data_original, data_imputed, column, idx_NA_test, type)
            scores.append(score)
        
        # Score moyen des 5 plis
        mean_score = np.mean(scores)
        
        # Mise à jour du meilleur score et des meilleurs paramètres si la performance est meilleure
        if (type == 'cat' and mean_score > best_score) or (type == 'num' and mean_score < best_score):
            best_score = mean_score
            best_params = params
            best_imputed_data = data_imputed.copy()        
    
    # Ré-entraînement de l'imputeur avec les meilleurs paramètres trouvés sur l'ensemble des données
    imputer = imputers[method](column, type, **best_params)

    # Démarrer la surveillance 
    tracemalloc.start() # de la mémoire
    start_time = time.time() # du temps

    # Imputer les données avec les meilleurs paramètres
    best_imputed_data = imputer.fit_transform(data_ampute)

    exec_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    mem_used = (peak - current) / (1024 * 1024)  # Convertir la mémoire en MB
    tracemalloc.stop()
    
    return best_score, exec_time, mem_used, best_imputed_data