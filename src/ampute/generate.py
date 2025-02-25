import numpy as np
from src.utils.data_utils import set_seed
from src.ampute.ampute_utils import mcar_amputation, sigmoid_amputation

def generate_ampute_mask(data: np.ndarray, column: int, mechanism: str, proportion: float, aux_cols: list = None, cols_names:list = None) -> np.ndarray:
    """
    Génère un masque de données manquantes selon les mécanismes MCAR, MAR, ou MNAR.

    Args:
        data: np.ndarray
            Données d'entrée.
        column: int
            Colonnes à amputer (indices).
        mechanism: str
            Mécanisme des données manquantes ('MCAR', 'MAR', 'MNAR').
        proportion: float
            Proportion cible de données manquantes (entre 0 et 1).
        aux_cols: list
            Liste des colonnes prédictives (indices), utilisée pour MAR et MNAR.
        cols_names: list
            Liste des noms des variables prédictives, utilisée pour MAR et MNAR.

    Returns:
        np.ndarray
            Masque des données manquantes (1 pour manquant, 0 pour présent).
    """
    if mechanism in ["MAR", "MNAR"]:
        if mechanism == "MAR": 
            gamma = 0
        else: gamma =  1
        missing_data_mask = sigmoid_amputation(
            data=data,
            column=column,
            proportion=proportion,
            aux_cols=aux_cols,
            gamma=gamma, 
            cols_names=cols_names
        )
    else: # MCAR
        missing_data_mask = mcar_amputation(data, column=column, proportion=proportion)

    # Validation finale du masque
    if missing_data_mask.shape != data.shape:
        raise ValueError("Le masque généré ne correspond pas à la forme des données d'entrée.")
    
    return missing_data_mask


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applique un masque de données manquantes à un tableau.

    Paramètres :
    - data : np.ndarray
        Données d'entrée (tableau numpy).
    - mask : np.ndarray
        Masque des données manquantes (1 pour manquant, 0 pour présent).

    Retour :
    - amputed_data : np.ndarray
        Tableau avec des valeurs manquantes (remplacées par np.nan).
    """
    # Applique le masque en remplaçant les éléments manquants par np.nan
    amputed_data = data.copy()
    amputed_data[mask == 1] = np.nan

    return amputed_data


def generate_ampute_np(data: np.ndarray, column: int, mechanism: str, proportion: float, aux_cols: list = None,
                         seed=None, cols_names:list = None) -> np.ndarray:
    """
    Introduit des données manquantes dans un tableau NumPy en fonction d'un mécanisme donné (MCAR, MAR, MNAR).

    Args:
        data: Ensemble de données complet sous forme de tableau NumPy sans valeurs manquantes.
        column: Indice de la colonne à rendre manquante.
        mechanism: Type de mécanisme de suppression ('MCAR', 'MAR', 'MNAR').
        proportion: Proportion des données à rendre manquantes.
        aux_cols: Liste des colonnes auxilliaires (nécessaire pour MAR et MNAR).
        cols_names: list
            Liste des noms des variables prédictives, utilisée pour MAR et MNAR.
    
    Returns:
        Un tableau NumPy avec les valeurs manquantes introduites selon le mécanisme spécifié.
    """
    if seed is not None:
        set_seed(seed)
        
    # Copie de data pour éviter la modification en place
    data_copy = data.copy()
    
    try:
        # Vérification du mécanisme choisi
        if mechanism not in ["MCAR", "MAR", "MNAR"]:
            raise ValueError("Le mécanisme doit être 'MCAR', 'MAR' ou 'MNAR'.")

        # Vérification de la proportion
        if not (0 < proportion <= 1):
            raise ValueError("La proportion doit être comprise entre 0 et 1.")

        # Vérification de la validité de la colonne
        if not (isinstance(column, int) and 0 <= column < data.shape[1]):
            raise ValueError("La colonne doit être un indice entier valide dans le tableau.")

        # Vérification de la validité des colonnes auxiliaires
        if aux_cols is not None:
            if not isinstance(aux_cols, list):
                raise ValueError("aux_cols doit être une liste d'indices de colonnes.")
            if not all(isinstance(col, int) and 0 <= col < data.shape[1] for col in aux_cols):
                raise ValueError("Toutes les colonnes dans aux_cols doivent être des indices entiers valides dans le tableau.")
            
        # Génération du masque de données manquantes pour la colonne spécifiée
        mask = generate_ampute_mask(
            data=data_copy,
            proportion=proportion,
            mechanism=mechanism,
            column=column,
            aux_cols=aux_cols,
            cols_names=cols_names
        )

        # Application du masque au tableau de données
        data_copy[:, column] = apply_mask(data=data_copy[:, column], mask=mask[:, column])

        return data_copy

    except Exception as e:
        print(f"Erreur lors de la génération des données manquantes : {e}")
        raise