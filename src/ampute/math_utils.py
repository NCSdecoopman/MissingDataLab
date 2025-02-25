import numpy as np
from scipy.optimize import root_scalar, minimize

def fit_intercepts(X, coeffs, p, self_mask=False):
    """
    Ajuste les intercepts pour obtenir une proportion cible `p` de valeurs manquantes.

    Paramètres :
    - X : np.ndarray, shape (n, d)
        Données d'entrée (n exemples, d dimensions).
    - coeffs : np.ndarray, shape (d,) ou (d_obs, d_na)
        Coefficients pour chaque dimension.
    - p : float
        Proportion cible de valeurs manquantes (entre 0 et 1).
    - self_mask : bool
        Si True, ajuste chaque dimension séparément.

    Retourne :
    - intercepts : np.ndarray
        Intercepts ajustés pour chaque dimension.
    """
    # S'assure que coeffs est une matrice 2D
    coeffs = np.atleast_2d(coeffs)

    # Ajuste la forme des coefficients si nécessaire
    if coeffs.shape[0] != X.shape[1]:
        coeffs = coeffs.T

    # Détermine le nombre d'intercepts à ajuster
    n_outputs = coeffs.shape[1] if not self_mask else len(coeffs)

    # Initialise les intercepts à zéro
    intercepts = np.zeros(n_outputs)

    # Parcourt chaque colonne ou dimension à ajuster
    for j in range(n_outputs):
        def f(x):
            # Calcule la différence entre proportion actuelle et cible
            if self_mask:
                probabilities = sigmoid(X @ coeffs[j] + x)
            else:
                probabilities = sigmoid(X @ coeffs[:, j] + x)
            return probabilities.mean() - p

        # Trouve l'intercept qui annule f(x) entre [-50, 50]
        try:
            result = root_scalar(f, method='brentq', bracket=[-50, 50], xtol=1e-8)
            if not result.converged:
                raise ValueError("L'optimisation n'a pas convergé.")
            intercepts[j] = result.root  # Stocke l'intercept trouvé
        except Exception as e:
            raise RuntimeError(f"Erreur pour l'interception {j} : {e}")

    # Retourne les intercepts ajustés
    return intercepts



def sigmoid(z):
    """
    Fonction sigmoïde : transforme une entrée en une probabilité entre 0 et 1.

    Paramètres :
    - z : np.ndarray ou float
        Entrée (peut être une seule valeur ou un tableau).
    
    Retour :
    - np.ndarray ou float
        Probabilité transformée entre 0 et 1.
    """
    
    return 1 / (1 + np.exp(-z))  



def objective(coeffs, X, target_proportion, alpha=0.01):
    """
    Fonction objective pour optimiser les coefficients en vue de produire une proportion cible de valeurs manquantes.

    Paramètres :
    - coeffs : np.ndarray, shape (d,)
        Coefficients à optimiser.
    - X : np.ndarray, shape (n, d)
        Données observées (n = échantillons, d = variables).
    - target_proportion : float
        Proportion cible de données manquantes (entre 0 et 1).
    - alpha : float
        Pénalité pour la régularisation.

    Retour :
    - error : float
        Erreur entre la proportion souhaitée et actuelle, avec régularisation.
    """
    # Calcule de z à partir des données et des coefficients.
    z = X @ coeffs   
        
    # Probabilité de données manquantes (sigmoïde).
    prob_missing = sigmoid(z)      
    
    # Proportion actuelle de valeurs manquantes.
    current_proportion = prob_missing.mean() 
    
    # Erreur entre proportion actuelle et cible. 
    error = (current_proportion - target_proportion) ** 2  
    
    # Pénalité pour limiter la taille des coefficients.  
    regularization = alpha * np.sum(coeffs ** 2)     

    # Retourne l'erreur totale.
    return error + regularization



def generate_coefficients(X, target_proportion):
    """
    Génère les coefficients optimisés pour atteindre une proportion cible.

    Paramètres :
    - X : np.ndarray, shape (n, d)
        Données observées.
    - target_proportion : float
        Proportion cible de données manquantes (entre 0 et 1).

    Retour :
    - coeffs : np.ndarray, shape (d,)
        Coefficients optimisés.
    """
    # le nombre de variables (colonnes).
    n, d = X.shape  
    
    # Initialise les coefficients aléatoirement.
    initial_coeffs = np.random.randn(d)   
    
    # Définition de la contrainte : 
    def constraint_lower(coeffs):
        return np.abs(coeffs) - 0.05  # Doit être ≥ 0 (|coeffs| > 0.05)

    def constraint_upper(coeffs):
        return 1 - np.abs(coeffs)  # Doit être ≥ 0 (|coeffs| < 1)

    constraints = [{'type': 'ineq', 'fun': constraint_lower},  # |coeffs| > 0.05
                {'type': 'ineq', 'fun': constraint_upper}]  # |coeffs| < 1

    # Minimise la fonction objective pour trouver les meilleurs coefficients.
    result = minimize(objective, initial_coeffs, args=(X, target_proportion), constraints=constraints, method='trust-constr')   

    # Retourne les coefficients optimisés.
    return result.x