import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

def set_seed(seed): # Définit la reproductibilité
    random.seed(seed)  # Affecte le générateur aléatoire de Python
    np.random.seed(seed)  # Affecte le générateur aléatoire de NumPy

def load_csv(file_path, delimiter=",", skip_header=0):
    """
    Charge un fichier CSV et retourne un DataFrame pandas.

    :param file_path: Chemin du fichier CSV.
    :param delimiter: Délimiteur utilisé dans le fichier (par défaut ',').
    :param skip_header: Nombre de lignes d'en-tête à ignorer (par défaut 0).
    :return: DataFrame pandas contenant les données du CSV.
    """
    df = pd.read_csv(file_path, delimiter=delimiter, skiprows=skip_header)
    print(f"Chargement du fichier {file_path}")
    return df

def save_csv(df, file_path, file_name, delimiter=",", index=False):
    """
    Sauvegarde un DataFrame pandas en fichier CSV.

    :param df: DataFrame pandas à sauvegarder.
    :param file_path: Chemin du fichier de destination.
    :param delimiter: Délimiteur à utiliser dans le fichier (par défaut ',').
    :param index: Booléen pour sauvegarder ou non l'index (par défaut False).
    """
    path_ampute_csv = os.path.join(file_path, f"{file_name}.csv")
    df.to_csv(path_ampute_csv, sep=delimiter, index=index)
    print(f"Fichier sauvegardé sous {path_ampute_csv}")

def encode_categorical_columns(df):
    """
    Encode les colonnes catégorielles d'un DataFrame avec Label Encoding.

    :param df: DataFrame pandas contenant les données à encoder.
    :return: DataFrame encodé et dictionnaire des encodeurs utilisés.
    """
    df_encoded = df.copy()  # Faire une copie pour éviter de modifier l'original
    label_encoders = {}  # Stocker les encodeurs pour une utilisation future

    # Identifier les colonnes catégorielles (de type 'object' ou 'category')
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_columns:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df[col])  # Encoder la colonne
        label_encoders[col] = encoder  # Sauvegarder l'encodeur

    return df_encoded, label_encoders

def decode_categorical_columns(df_encoded, label_encoders):
    """
    Décode un DataFrame encodé avec Label Encoding pour retrouver les valeurs d'origine.

    :param df_encoded: DataFrame encodé.
    :param label_encoders: Dictionnaire des encodeurs utilisés pour chaque colonne.
    :return: DataFrame avec les valeurs originales restaurées.
    """
    df_decoded = df_encoded.copy()  # Copier pour éviter de modifier l'original

    for col, encoder in label_encoders.items():
        # Stocker la position des NaN
        nan_mask = df_encoded[col].isna()

        # Si la colonne contient des NaN, on les remplace temporairement par la valeur la plus fréquente
        if nan_mask.any():
            most_frequent_value = df_encoded[col].mode()[0]  # Trouve la valeur la plus fréquente
            temp_col = df_encoded[col].fillna(most_frequent_value).astype(int)  # Remplace NaN et cast en int
        else:
            temp_col = df_encoded[col].astype(int)

        # Décoder les valeurs
        df_decoded[col] = encoder.inverse_transform(temp_col)

        # Restaurer les NaN après le décodage
        df_decoded.loc[nan_mask, col] = np.nan

    return df_decoded

def convert_dataframe_to_numpy(df_pd):
    """
    Convertit un DataFrame pandas en array NumPy et conserve la correspondance index-colonnes.

    :param df_pd: DataFrame pandas à convertir.
    :return: Tuple (array NumPy, dictionnaire {index colonne : nom colonne})
    """
    df_np = df_pd.to_numpy()  # Convertir en array NumPy
    column_mapping = {i: col for i, col in enumerate(df_pd.columns)}  # Associer index -> nom colonne

    return df_np, column_mapping

def convert_numpy_to_dataframe(df_np, column_mapping):
    """
    Convertit un array NumPy en DataFrame pandas en rétablissant les noms des colonnes.

    :param df_np: Array NumPy contenant les données.
    :param column_mapping: Dictionnaire {index colonne : nom colonne}.
    :return: DataFrame pandas restauré.
    """
    # Récupérer les noms des colonnes en respectant l'ordre des index
    column_names = [column_mapping[i] for i in range(df_np.shape[1])]

    # Convertir en DataFrame avec les colonnes restaurées
    df_pd = pd.DataFrame(df_np, columns=column_names)

    return df_pd

def save_results(results, dir_path, file_name):
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    save_csv(results_df, file_path=dir_path, file_name=file_name)

def parse_and_aggregate(series):
    """
    Agrège les valeurs d'une série de listes en une seule liste et calcule la moyenne et l'écart-type.
    """
    all_values = [value for lst in series for value in lst]
    return np.mean(all_values), np.std(all_values)

def aggregate_grouped_data(df):
    """
    Regroupe les données selon les colonnes spécifiées et agrège les colonnes numériques.
    """
    df = pd.DataFrame(df)
    # Groupement selon les colonnes spécifiées
    grouped = df.groupby(['col_name', 'type', 'mechanism', 'proportion', 'imputation_method'])

    aggregated_data = []
    for name, group in grouped:
        scores_mean, scores_std = parse_and_aggregate(group['scores'])
        exec_times_mean, exec_times_std = parse_and_aggregate(group['exec_times'])
        mem_usages_mean, mem_usages_std = parse_and_aggregate(group['mem_usages'])
        variances_mean, variances_std = parse_and_aggregate(group['variances'])

        aggregated_data.append(list(name) + [scores_mean, scores_std, exec_times_mean, exec_times_std,
                                             mem_usages_mean, mem_usages_std, variances_mean, variances_std])

    # Création du DataFrame final
    columns = ['col_name', 'type', 'mechanism', 'proportion', 'imputation_method',
               'score_mean', 'score_std', 'exec_time_mean', 'exec_time_std',
               'mem_used_mean', 'mem_used_std', 'variance_mean', 'variance_std']

    result_df = pd.DataFrame(aggregated_data, columns=columns)
    return result_df

def aggregate_grouped_variances(df):
    """
    Regroupe les données selon 'col_name' et 'mechanism'
    et calcule la moyenne des proportions, puis la moyenne
    et l'écart-type de la variance dans chaque groupe.

    Retourne la liste de tuples :
        (col_name, mechanism, proportion_mean, variance_mean, variance_std)
    """
    df = pd.DataFrame(df)
    
    grouped_df = df.groupby(['col_name', 'mechanism', 'proportion'], as_index=False).agg(
        variance_mean=('variance', 'mean'),
        variance_std=('variance', 'std')
    )
    
    return grouped_df
