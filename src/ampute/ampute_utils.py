import numpy as np

from src.ampute.math_utils import generate_coefficients, fit_intercepts, sigmoid

def mcar_amputation(data: np.ndarray, column: int, proportion: float) -> np.ndarray:
    """
    Génère un masque aléatoire pour simuler des données manquantes sur des colonnes spécifiées.

    Paramètres :
    - data : np.ndarray
        Données d'entrée (tableau numpy).
    - column : int
        Indices de colonne à amputer.
    - proportion : float
        Proportion cible de données manquantes (entre 0 et 1).

    Retour :
    - missing_data_mask : np.ndarray
        Masque des données manquantes (1 pour manquant, 0 pour présent).
    """ 
    # Copie des données pour ne pas modifier l'original
    missing_data_mask = np.zeros_like(data, dtype=int)
    
    # Appliquer le masque uniquement sur la colonne donnée
    missing_data_mask[:, column] = (np.random.rand(data.shape[0]) < proportion).astype(int)
    
    return missing_data_mask


def sigmoid_amputation(data: np.ndarray, column: int, proportion: float, aux_cols: list, 
                       gamma: int = None, cols_names: list = None, activate_pattern: bool = True) -> np.ndarray:
    """
    Simule des données MNAR selon des prédicteurs et une méthode spécifiée.

    Paramètres:
    - data : np.ndarray, shape (n, d)
        Données d'entrée.
    - column: int
        Indice de colonne à rendre manquante.
    - proportion: float
        Proportions de valeurs manquantes pour chaque colonne cible.
    - aux_cols: list
        Colonnes auxilliaires.
    - gamma : int
        Indique le mécanisme : Si 0 : MAR, sinon MNAR.
    cols_names: list
            Liste des noms des variables prédictives, utilisée pour controler le "sens" des effets des variables

    Retour :
    - mask : np.ndarray, shape (n, d)
        Masque des données manquantes (1 pour manquant, 0 sinon).
    """    
    # Vérification des contraintes sur aux_cols en fonction de gamma
    if gamma == 0:
        if aux_cols is not None and column in aux_cols:
            raise ValueError("Erreur : aux_cols ne doit pas contenir column lorsque gamma = 0.")

    elif gamma == 1:
        if aux_cols is None or column not in aux_cols:
            raise ValueError("Erreur : aux_cols doit contenir column lorsque gamma = 1.")

    n, d = data.shape
    data = data[:, aux_cols].astype(float)

    # Génère des coefficients optimisés pour atteindre la proportion cible
    optimized_coeffs = generate_coefficients(data, proportion)

    # Ajustement des coefficients en fonction des colonnes disponibles suivant des patterns choisis
    # Ici relations logiques spécifiques entre les variables (âge, tabagisme, accès à l'aide)
    def set_coefficients(var1, var2, coeff_neg, coeff_pos):
        try:
            idx_var1 = cols_names.index(var1)
            idx_var2 = cols_names.index(var2)
            return [coeff_neg, coeff_pos] if aux_cols[idx_var1] < aux_cols[idx_var2] else [coeff_pos, coeff_neg]
        except ValueError as e:
            raise ValueError(f"Erreur lors de la recherche des variables : {e}")

    if activate_pattern:
        # Dictionnaire des configurations possibles pour les paires de colonnes
        patterns = {
            ("Age_Group", "Smoking_Prevalence"): (-0.5, 0.07),
            ("Access_to_Counseling", "Smoking_Prevalence"): (0.6, 0.07)
        }

        # Application du pattern approprié
        for (var1, var2), (coeff_neg, coeff_pos) in patterns.items():
            if var1 in cols_names and var2 in cols_names:
                optimized_coeffs[:2] = set_coefficients(var1, var2, coeff_neg, coeff_pos)
                break
        else:
            # Si aucun pattern ne correspond, appliquer des valeurs par défaut
            if "Age_Group" in cols_names:
                optimized_coeffs[0] = -0.5
            elif "Access_to_Counseling" in cols_names:
                optimized_coeffs[0] = 0.6
            else:
                optimized_coeffs[0] = 0.07
 

    # Ajuste les intercepts pour atteindre précisément la proportion cible
    intercepts = fit_intercepts(data, optimized_coeffs, proportion)

    # Assure que les dimensions de intercepts sont compatibles
    intercepts = np.array(intercepts).reshape(1, -1)  # Convertit en (1, d)

    # Calcule les probabilités d'être manquant avec les intercepts ajustés
    z = (data @ optimized_coeffs).reshape(-1, 1) + intercepts
    prob_missing = sigmoid(z)

    # Génère un masque aléatoire pour chaque cellule
    mask = np.zeros((n, d))
    random_mask = np.random.rand(n, 1)
    mask[:, column] = (random_mask < prob_missing).flatten()

    # Convertit le masque en entier (0 ou 1)
    return mask.astype(int)