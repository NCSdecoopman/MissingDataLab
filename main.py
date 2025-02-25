import yaml
import numpy as np
import sys

from joblib import Parallel, delayed

from src.utils.path_utils import create_directories

from src.utils.data_utils import load_csv, save_results
from src.utils.data_utils import encode_categorical_columns, convert_dataframe_to_numpy

from src.ampute.generate import generate_ampute_np
from src.impute.generate import generate_impute_np

from src.utils.data_utils import aggregate_grouped_data, aggregate_grouped_variances

from src.utils.graph_utils import plot_imputation_results

def load_config(file_path):
    """Charge le fichier YAML de configuration."""
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def main(config):
    # Créer les répertoires nécessaires
    create_directories(config)

    # Définition des répertoires
    path_result_csv = config["result"]["csv_directory"]
    path_result_jpg = config["result"]["jpg_directory"]
    
    # CHARGEMENT DU FICHIER ORIGINAL
    data_origin_pd = load_csv(
        config["data_origin"]["csv_file"],
        config["data_origin"]["delimiter"],
        config["data_origin"]["skip_header"]
    )

    # CHARGEMENT DES CONFIGURATIONS
    proportions = config["missing_data"]["proportion"]
    seed = config["seed"]
    constant_value = config.get("missing_data", {}).get("constante", None)
    evaluation_amputation = config["evaluation"]["repetitions_amputation"]
    evaluation_imputation = config["evaluation"]["repetitions_imputation"]

    # Récupérer la liste (ou None)
    columns_cat = config["missing_data"].get("columns_cat", [])
    columns_num = config["missing_data"].get("columns_num", [])

    # Convertir en liste vide si c'est None
    if columns_cat is None:
        columns_cat = []
    if columns_num is None:
        columns_num = []

    # Construction de la liste col en gérant les catégories et les numériques
    col = (
        [{"cat": col_data.pop("column"), **col_data} for col_data in columns_cat] +
        [{"num": col_data.pop("column"), **col_data} for col_data in columns_num]
    )

    # Vérifier s'il y a des valeurs manquantes dans le jeu de données d'origine
    if data_origin_pd.isna().any().any():
        print("ERREUR : Le fichier CSV contient des valeurs manquantes. Le programme nécessite un jeu de données complet.")
        sys.exit(1)

    # Initialisation du stockage des résultats
    results = []  
    variance_originale = []

    # ENCODAGE DES VARIABLES CATEGORIELLES
    data_origin_pd_encode, label_encoders = encode_categorical_columns(data_origin_pd)

    # TRANSFORMATION EN NUMPY ARRAY
    data_np, column_mapping_np = convert_dataframe_to_numpy(data_origin_pd_encode)

    # Calcul du nombre total d'itérations pour l'amputation
    total_amputation_iterations = (
        evaluation_amputation
        * len(proportions)
        * sum(len(item["mechanisms"]) for item in col)
    )
    amputation_count = 0

    # Boucle sur le nombre de données manquantes à générer
    for i in range(evaluation_amputation):
        # On modifie la graine à chaque itération
        iteration_seed = seed + i if seed is not None else np.random.randint(10**9)

        # Boucle sur les proportions de données manquantes
        for proportion in proportions:
            # Boucle sur chaque colonne
            for item in col:
                col_type = "cat" if "cat" in item else "num"
                col_name = item[col_type]
                # Trouver l'indice de la colonne
                col_index = next(key for key, value in column_mapping_np.items() if value == col_name)

                # Vérifier si la colonne est catégorielle et encoder constant_value si nécessaire
                if col_type == "cat" and constant_value is not None:
                    if col_name in label_encoders:
                        try:
                            constant_value_encoded = label_encoders[col_name].transform([constant_value])[0]
                        except ValueError:
                            print(
                                f"Attention : La valeur constante '{constant_value}' "
                                f"n'est pas présente dans l'encodeur pour la colonne '{col_name}'."
                            )
                            constant_value_encoded = None
                    else:
                        constant_value_encoded = None
                else:
                    constant_value_encoded = constant_value

                # Boucle sur chaque mécanisme d'amputation
                for mechanism in item["mechanisms"]:
                    amputation_count += 1
                    print(
                        f"[Amputation {amputation_count}/{total_amputation_iterations}] "
                        f"proportion: {proportion}, colonne: {col_name}, mécanisme: {mechanism}"
                    )

                    try:
                        if mechanism in ["MCAR"]:
                            aux_cols = None
                        elif mechanism in ["MAR"]:
                            aux_cols = item["mar_auxiliary_columns"]
                        elif mechanism in ["MNAR"]:
                            aux_cols = item["mar_auxiliary_columns"] + [col_name]
                        else:
                            raise ValueError("Le mécanisme doit être 'MCAR', 'MAR' ou 'MNAR'.")

                        # Récupérer les indices des colonnes auxiliaires
                        if aux_cols is not None:
                            aux_index = [key for key, value in column_mapping_np.items() if value in aux_cols]
                            col_names = list(data_origin_pd.columns[aux_index])
                        else:
                            aux_index = None  # Si aux_cols est None, on garde None
                            col_names = None
                            
                        # GENERATION DES DONNEES MANQUANTES
                        data_ampute_np = generate_ampute_np(
                            data=data_np,
                            column=col_index,
                            mechanism=mechanism,
                            proportion=proportion,
                            aux_cols=aux_index,
                            seed=iteration_seed,
                            cols_names=col_names
                        )

                        # Calcul de la variance sur les valeurs amputées
                        idx_NA = np.where(np.isnan(data_ampute_np[:, col_index]))[0]
                        var_ampute = np.var(data_np[idx_NA, col_index], ddof=0)
                        variance_originale.append({
                            "col_name": col_name,
                            "mechanism": mechanism,
                            "proportion": proportion,
                            "seed": iteration_seed,
                            "variance": var_ampute
                        })

                        # PROCESSUS D'IMPUTATION
                        for method in item["imputation_methods"]:
                            print(
                                f"    Méthode d'imputation : {method} "
                                f"(répétitions: {evaluation_imputation})"
                            )
                            try:
                                # Exécution parallèle des répétitions d'imputation
                                results_list = Parallel(n_jobs=-1)(
                                    delayed(generate_impute_np)(
                                        data_original=data_np,
                                        data_ampute=data_ampute_np,
                                        column=col_index,
                                        type=col_type,
                                        method=method,
                                        constant_value=constant_value_encoded
                                    ) 
                                    for _ in range(evaluation_imputation)
                                )

                                # Récupération et agrégation des résultats
                                scores = []
                                exec_times = []
                                mem_usages = []
                                variances = []

                                for (data_impute_np, score, exec_time, mem_used, variance) in results_list:
                                    scores.append(float(score))
                                    exec_times.append(float(exec_time))
                                    mem_usages.append(float(mem_used))
                                    variances.append(float(variance))

                                results.append({
                                    "col_name": col_name,
                                    "type": col_type,
                                    "mechanism": mechanism,
                                    "proportion": proportion,
                                    "seed": iteration_seed,
                                    "imputation_method": method,
                                    "scores": scores,
                                    "exec_times": exec_times,
                                    "mem_usages": mem_usages,
                                    "variances": variances
                                })

                            except Exception as e:
                                print(
                                    f"Erreur lors du traitement de '{col_name}' "
                                    f"avec '{mechanism}' (méthode: {method}): {e}"
                                )

                    except Exception as e:
                        print(f"Erreur lors de l'amputation de '{col_name}' avec '{mechanism}': {e}")

    # Agrégation des données
    results_agregate = aggregate_grouped_data(results)
    variance_originale_agregate = aggregate_grouped_variances(variance_originale)

    # Sauvegarde des résultats
    save_results(results_agregate, path_result_csv, file_name="results_score")
    save_results(variance_originale_agregate, path_result_csv, file_name="variance_originale")

    # Génération du graphique
    plot_imputation_results(results_agregate, variance_originale_agregate, path_result_jpg, file_name="results_graph")

if __name__ == "__main__":
    config_path = "config/config.yaml"
    config_data = load_config(config_path)
    main(config_data)