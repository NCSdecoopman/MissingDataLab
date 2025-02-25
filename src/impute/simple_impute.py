import numpy as np
import tracemalloc
import time

from src.utils.metrics_utils import score_imputed

class BaseImputer:
    def __init__(self, column, type, **kwargs):
        self.column = column
        self.type = type
        self.kwargs = kwargs

    def fit_transform(self, X):
        raise NotImplementedError("Chaque classe doit implémenter fit_transform")

class SimpleImputerMethod(BaseImputer):    
    def fit_transform(self, X):
        return self.simple_impute(X, self.column, **self.kwargs)
    
    def simple_impute(self, data_ampute, column, method, constant_value=None):
        col_values = data_ampute[:, column]
        
        if method == 'mean':
            imputed_value = np.nanmean(col_values)
        elif method == 'median':
            imputed_value = np.nanmedian(col_values)
        elif method == 'mode':
            unique_values, counts = np.unique(col_values[~np.isnan(col_values)], return_counts=True)
            imputed_value = unique_values[np.argmax(counts)] if unique_values.size > 0 else np.nan
        elif method == 'constante':
            if constant_value is None:
                raise ValueError("Une constante doit être fournie pour l'imputation constante.")
            imputed_value = constant_value
        else:
            raise ValueError("Méthode d'imputation non valide. Utilisez 'mean', 'median', 'mode' ou 'constante'.")
        
        data_transformed = data_ampute.copy()
        col_mask = np.isnan(data_transformed[:, column])
        
        # Test des performances

        # Démarrer la surveillance 
        tracemalloc.start() # de la mémoire
        start_time = time.time() # du temps

        # Imputer
        data_transformed[:, column][col_mask] = imputed_value

        exec_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        mem_used = (peak - current) / (1024 * 1024)  # Convertir la mémoire en MB
        tracemalloc.stop()
        
        return data_transformed, exec_time, mem_used

def simple_imputation(data_original: np.ndarray, data_ampute: np.ndarray, column: int, idx_NA: list, 
                      type: str, method: str, constant_value=None):                      
    """
    Fonction pour effectuer une imputation simple des valeurs manquantes dans une colonne spécifique d'un tableau de données.

    data_original : tableau de données original sans valeurs manquantes (utilisé pour évaluer la qualité de l'imputation).
    data_ampute : tableau de données avec des valeurs manquantes (à imputer).
    column : index de la colonne à imputer.
    idx_NA : liste des indices des lignes où la colonne spécifiée contient des valeurs manquantes.
    type : type de données (par exemple, "numérique", "catégorique") qui pourrait influencer la méthode d'imputation.
    method : str
        Méthode d'imputation choisie parmi ["mean", "median", "mode", "constante"] :
            - "mean" : imputation par la moyenne
            - "median" : imputation par la médiane
            - "mode" : imputation par la valeur la plus fréquente
            - "constante" : imputation par une valeur constante définie par l'utilisateur
    constant_value : valeur constante à utiliser pour l'imputation si la méthode "constante" est choisie.

    Retourne :
    ----------
    final_score : score mesurant la qualité de l'imputation (par exemple RMSE ou MAE selon l'implémentation de `score_imputed`).
    exec_time : temps d'exécution de l'imputation.
    mem_used : mémoire utilisée pendant l'imputation.
    best_imputed_data : données après imputation.
    """

    # Dictionnaire mappant les méthodes d'imputation aux implémentations correspondantes
    imputers = {
        "mean": SimpleImputerMethod,      # Imputation par la moyenne
        "median": SimpleImputerMethod,    # Imputation par la médiane
        "mode": SimpleImputerMethod,      # Imputation par la valeur la plus fréquente
        "constante": SimpleImputerMethod  # Imputation par une valeur constante
    }

    # Initialisation de l'imputeur en fonction de la méthode choisie
    imputer = imputers[method](column, type, method=method, constant_value=constant_value)

    # Application de l'imputation et récupération du résultat, du temps d'exécution et de la mémoire utilisée
    best_imputed_data, exec_time, mem_used = imputer.fit_transform(data_ampute)
    
    # Évaluation de la qualité de l'imputation par comparaison avec les données originales
    final_score = score_imputed(data_original, best_imputed_data, column, idx_NA, type)
    
    return final_score, exec_time, mem_used, best_imputed_data