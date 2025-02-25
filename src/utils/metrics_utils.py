import numpy as np

def score_imputed(data_original: np.ndarray, data_imputed: np.ndarray, column: int, idx_na: list, type: str) -> float:
    """
    Calcule l'accuracy pour une colonne catégorielle ou la MSE pour une colonne numérique.
    
    :param data_original: np.ndarray, tableau original avec valeurs non imputées
    :param data_imputed: np.ndarray, tableau avec valeurs imputées
    :param column: int, indice de la colonne à analyser
    :param idx_na: list, indice des individus imputés
    :param type: str, 'cat' pour une variable catégorielle, 'num' pour une variable numérique
    :return: float, accuracy (%) si 'cat', MSE si 'num'
    """
    # Vérifications des entrées
    if type not in ['cat', 'num']:
        raise ValueError("Le paramètre 'type' doit être soit 'cat' (catégoriel) soit 'num' (numérique).")

    if len(idx_na) == 0:
        raise ValueError("La liste 'idx_na' est vide. Aucune valeur imputée à évaluer.")

    # Sélection des valeurs originales et imputées
    original_values = data_original[idx_na, column]
    imputed_values = data_imputed[idx_na, column]

    if type == 'cat':
        # Calcul de l'accuracy pour les variables catégorielles
        accuracy = np.mean(np.equal(original_values, imputed_values)) * 100
        return accuracy

    elif type == 'num':
        # Calcul du MSE pour les variables numériques
        mse = np.mean((original_values - imputed_values) ** 2)
        return mse
    
def variance_imputed(data_imputed: np.ndarray, column: int, idx_na: list) -> float:
    # Calcule la variance des valeurs imputées pour une colonne spécifique dans un tableau de données.
    imputed_values = data_imputed[idx_na, column]
    return np.var(imputed_values)