import numpy as np

from src.impute.simple_impute import simple_imputation
from src.impute.advance_impute import advance_imputation

from src.utils.metrics_utils import variance_imputed

def generate_impute_np(
    data_original: np.ndarray, 
    data_ampute: np.ndarray, 
    column: int, 
    type: str, 
    method: str, 
    constant_value=None
) -> tuple[np.ndarray, float, float, float, float]:
    """
    Impute des données manquantes dans un tableau NumPy en fonction d'une méthode.

    Args:
        data_original (np.ndarray): Ensemble de données complet sans valeurs manquantes.
        data_ampute (np.ndarray): Ensemble de données incomplet avec des valeurs manquantes.
        column (int): Indice de la colonne à imputer.
        type (str): Type de variable ('cat' pour catégorielle, 'num' pour numérique).
        method (str): Méthode d'imputation.
        constant_value (optional): Valeur utilisée pour l'imputation constante (nécessaire si method='constante').

    Returns:
        tuple: (data_imputed, score, exec_time, mem_used, variance)
    """

    # Copie de data pour éviter la modification en place
    data_imputed = data_ampute.copy()
    idx_NA = np.where(np.isnan(data_ampute[:, column]))[0]

    try:
        # Vérification du mécanisme choisi
        methods_supported = ["mean", "median", "mode", "constante", "KNN", "SoftImputer", "ICE", "MissForest", "ACP"]

        if method not in methods_supported:
            raise ValueError(f"La méthode d'imputation '{method}' n'est pas supportée. Méthodes valides : {methods_supported}")

        # Vérification du type
        if method in ["mean", "median", "SoftImputer", "ACP"] and type == "cat":
            raise ValueError(f"La méthode d'imputation '{method}' n'est pas appropriée pour les variables catégorielles.")

        if method == "mode" and type == "num":
            raise ValueError(f"La méthode d'imputation '{method}' n'est pas appropriée pour les variables numériques.")

        # Vérification de la validité de la colonne
        if not (isinstance(column, int) and 0 <= column < data_imputed.shape[1]):
            raise ValueError(f"La colonne doit être un indice entier valide compris entre 0 et {data_imputed.shape[1]-1}.")

        # Imputation simple
        if method in ["mean", "median", "mode", "constante"]:
            score, exec_time, mem_used, data_imputed = simple_imputation(
                data_original, data_ampute, column, idx_NA, type, method, constant_value
            )

        # Imputation avancée
        elif method in ["KNN", "SoftImputer", "ICE", "MissForest", "ACP"]:
            score, exec_time, mem_used, data_imputed = advance_imputation(
                data_original, data_ampute, column, idx_NA, type, method
            )

        # Calcul de la variance après imputation
        variance = variance_imputed(data_imputed, column, idx_NA)

        return data_imputed, score, exec_time, mem_used, variance

    except Exception as e:
        print(f"Erreur lors de la génération des données imputées : {e}")
        raise