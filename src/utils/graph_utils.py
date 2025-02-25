import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_imputation_results(df, variance, output_path, file_name):
    """
    Génère et enregistre des graphiques pour évaluer les résultats d'imputation de données manquantes 
    selon différentes méthodes et métriques.

    Paramètres :
    ------------
    df : pandas.DataFrame
        DataFrame contenant les résultats des imputations, incluant les métriques suivantes :
        - score_mean, score_std : Moyenne et écart-type du score (précision ou MSE).
        - exec_time_mean, exec_time_std : Moyenne et écart-type du temps d'exécution.
        - mem_used_mean, mem_used_std : Moyenne et écart-type de l'utilisation mémoire.
        - variance_mean, variance_std : Moyenne et écart-type de la variance des imputations.
        - col_name : Nom de la variable imputée.
        - imputation_method : Méthode d'imputation utilisée.
        - mechanism : Mécanisme des données manquantes (MCAR, MAR, MNAR).
        - proportion : Proportion de données manquantes.
        - type : Type de variable (catégorielle 'cat' ou numérique 'num').

    variance : pandas.DataFrame
        DataFrame contenant les informations sur la variance originale des données pour chaque variable 
        et mécanisme.

    output_path : str
        Chemin du répertoire où le graphique final sera sauvegardé.

    file_name : str
        Nom du fichier de sortie pour le graphique (sans extension).

    Étapes du traitement :
    ----------------------
    1. **Fusion et renommage :**
        - Fusionne les DataFrames `df` et `variance` pour enrichir les informations.
        - Renomme les colonnes pour des noms plus explicites et simplifiés.
        - Renomme les variables et les méthodes d'imputation pour des affichages plus lisibles.

    2. **Préparation des données :**
        - Transformation des données en format "long" pour faciliter l'utilisation de seaborn.
        - Ajout des barres d'erreurs associées à chaque métrique.
        - Création de colonnes combinées pour distinguer les métriques selon leur type (Score, Temps, Mémoire, Variance).
        - Suppression des valeurs non pertinentes (ex : Variance pour les variables catégorielles).

    3. **Visualisation :**
        - Création d'une grille de graphiques (`FacetGrid`) avec seaborn pour afficher les métriques selon :
            * Les types de variables et mécanismes des données manquantes.
            * Les méthodes d'imputation.
        - Tracé des courbes principales et des barres d'erreurs.
        - Ajout de la variance originale en fond pour comparaison.
        - Personnalisation des axes, légendes et étiquettes.

    4. **Enregistrement :**
        - Sauvegarde du graphique final sous le format PNG à l'emplacement spécifié.

    Résultat :
    ----------
    - Un fichier image contenant plusieurs graphiques comparant les performances des méthodes 
      d'imputation selon différentes métriques (Précision/MSE, Temps, Mémoire, Variance).
    - Le graphique inclut des lignes de référence pour la variance originale.
    """

    variance = variance.merge(df[['col_name', 'type']], on="col_name", how='left')

    # Renommer les colonnes pour simplifier l'affichage
    df = df.rename(columns={
        "score_mean": "Score",
        "score_std": "Score_std",
        "exec_time_mean": "Temps",
        "exec_time_std": "Temps_std",
        "mem_used_mean": "Mémoire",
        "mem_used_std": "Mémoire_std",
        "variance_mean": "Variance",
        "variance_std": "Variance_std"
    })

    # Renommer les colonnes
    rename_col_name = {
        "Access_to_Counseling": "Variable binaire",
        "Smoking_Prevalence": "Variable continue"
    }
    df["col_name"] = df["col_name"].map(rename_col_name)
    variance["col_name"] = variance["col_name"].map(rename_col_name)

    # Renommer les méthodes d'imputation
    rename_methods = {
        "constante": "Constante",
        "mode": "Mode",
        "mean": "Moyenne",
        "median": "Médiane",
        "KNN": "KNN",
        "SoftImputer": "SoftImputer",
        "ACP": "ACP",
        "ICE": "ICE",
        "MissForest": "MissForest"
    }
    df["imputation_method"] = df["imputation_method"].map(rename_methods)

    # Préparer les données pour le plotting
    melted_df = pd.melt(
        df,
        id_vars=["col_name", "type", "mechanism", "proportion", "imputation_method"],
        value_vars=["Score", "Temps", "Mémoire", "Variance"],
        var_name="Métrique",
        value_name="Valeur"
    )

    # Associer les colonnes des barres d'erreurs
    std_mapping = {
        "Score": "Score_std",
        "Temps": "Temps_std",
        "Mémoire": "Mémoire_std",
        "Variance": "Variance_std"
    }

    melted_df["Erreur_std"] = melted_df.apply(
        lambda row: df.loc[
            (df["col_name"] == row["col_name"]) &
            (df["mechanism"] == row["mechanism"]) &
            (df["proportion"] == row["proportion"]) &
            (df["imputation_method"] == row["imputation_method"]),
            std_mapping[row["Métrique"]]
        ].values[0], axis=1
    )

    # Créer une colonne combinée pour les lignes
    melted_df["var_mechanism"] = melted_df["col_name"] + " (" + melted_df["mechanism"] + ")"

    def combined_metric_type(row):
        """Crée un label séparé pour le Score selon qu'il s'agit de cat ou de num."""
        if row["Métrique"] == "Score":
            return "Score (Précision)" if row["type"] == "cat" else "Score (MSE)"
        else:
            return row["Métrique"]

    melted_df["metric_type"] = melted_df.apply(combined_metric_type, axis=1)

    # Remplacer les valeurs par NaN pour les catégorielles
    melted_df.loc[(melted_df['type'] == 'cat') & (melted_df['Métrique'] == 'Variance'), ['Valeur', 'Erreur_std']] = np.nan
    variance.loc[(variance['type'] == 'cat'), ['variance_mean', 'variance_std']] = np.nan

    # Définir les labels des axes Y
    y_labels = {
        "Score (Précision)": "Précision",
        "Score (MSE)": "MSE",
        "Temps": "Temps d'exécution (sec)",
        "Mémoire": "Pic de mémoire (MB)",
        "Variance": "Variance des imputations"
    }

    sns.set(style="white")

    g = sns.FacetGrid(
        melted_df,
        row="metric_type",            
        col="var_mechanism",
        hue="imputation_method",
        hue_order=["Constante", "Mode", "Moyenne", "Médiane", "KNN", "SoftImputer", "ACP", "ICE", "MissForest"],
        col_order=[f"{var} ({mech})" for var in melted_df["col_name"].unique() for mech in ["MCAR", "MAR", "MNAR"]],
        palette="colorblind",
        margin_titles=True,
        sharey="row",                 
        sharex=True,
        despine=False
    )
    g.set_titles(row_template="", col_template="{col_name}")

    g.map_dataframe(
        sns.lineplot,
        x="proportion",
        y="Valeur",
        errorbar=None
    )

    g.map_dataframe(
        plt.errorbar,
        "proportion",
        "Valeur",
        yerr="Erreur_std",
        fmt='o',
        capsize=0,
        markersize=3
    )

    def add_variance_points(data, **kwargs):
        if data["metric_type"].iloc[0] == "Variance":
            merged = data.merge(
                variance[["col_name", "mechanism", "proportion", "variance_mean", "variance_std"]],
                on=["col_name", "mechanism", "proportion"],
                how="left"
            )
            merged = merged.sort_values("proportion")

            line, = plt.plot(
                merged["proportion"],
                merged["variance_mean"],
                color='black',
                linestyle='-'
            )

            band = plt.fill_between(
                merged["proportion"],
                merged["variance_mean"] - merged["variance_std"],
                merged["variance_mean"] + merged["variance_std"],
                color='black',
                alpha=0.05
            )

            return line, band
        return None, None

    g.map_dataframe(add_variance_points)

    for ax, row_name in zip(g.axes[:, 0], g.row_names):
        ax.set_ylabel(y_labels[row_name])

    for ax in g.axes.flat:
        ax.set_xlabel("Proportion NaN")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_ylim(bottom=0)

    g.add_legend(title="Méthode d'imputation")
    handle = mlines.Line2D([], [], color='black', linestyle='-', label='Variance originale')
    g.fig.legend(handles=[handle], labels=['Variance originale'], loc='lower right', ncol=1)

    output = f"{output_path}/{file_name}.png"
    plt.savefig(output, bbox_inches='tight')
    print(f"Le graphique a été enregistré sous : {output}")
    plt.close()