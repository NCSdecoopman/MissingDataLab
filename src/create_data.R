# --- Initialisation ------------------------------------------------------------

library(simstudy)
library(data.table)

setwd("D:/Master/S9/Projet_NA/donnees_manquantes")
dir.create("data", showWarnings = FALSE) # Crée le dossier data s'il n'existe pas

# --- Fonction utilitaire pour le nettoyage des colonnes continues -------------

clean_continuous_vars <- function(dt, cols, min_val = 0, max_val = 10, round_digits = 0) {
  for (col in cols) {
    dt[get(col) < min_val, (col) := min_val]
    dt[get(col) > max_val, (col) := max_val]
    dt[, (col) := round(get(col), round_digits)]
  }
}

# --- Création du dataset avec dépendances (corrélations) ----------------------

cor_matrix <- matrix(
  c(1.0, 0.8, 0.4, 0.6, 0.6, 0.4, 0.5, 0.6, 0.5,
    0.8, 1.0, 0.5, 0.6, 0.5, 0.6, 0.4, 0.4, 0.5,
    0.4, 0.5, 1.0, 0.5, 0.6, 0.4, 0.5, 0.4, 0.4,
    0.6, 0.6, 0.5, 1.0, 0.4, 0.5, 0.6, 0.4, 0.5,
    0.6, 0.5, 0.6, 0.4, 1.0, 0.4, 0.4, 0.6, 0.6,
    0.4, 0.6, 0.4, 0.5, 0.4, 1.0, 0.6, 0.4, 0.5,
    0.5, 0.4, 0.5, 0.6, 0.4, 0.6, 1.0, 0.5, 0.4,
    0.6, 0.4, 0.4, 0.4, 0.6, 0.4, 0.5, 1.0, 0.6,
    0.5, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.6, 1.0), 
  nrow = 9, byrow = TRUE
)

var_names <- c("Smoking_Prevalence", "Drug_Experimentation", "Peer_Influence",
               "Family_Background", "Mental_Health", "Parental_Supervision",
               "Substance_Education", "Community_Support", "Media_Influence")

colnames(cor_matrix) <- var_names
rownames(cor_matrix) <- var_names

dt <- as.data.table(
  genCorData(
    n = 1000,
    mu = c(25, 10, rep(5, 7)),
    sigma = c(12, 4, rep(2, 7)),
    corMatrix = cor_matrix,
    cnames = var_names
  )
)

# --- Nettoyage des variables continues ----------------------------------------

# Plafonnement et arrondi des variables
dt[Smoking_Prevalence < 0, Smoking_Prevalence := 0]
dt[Drug_Experimentation < 0, Drug_Experimentation := 0]
dt[, Smoking_Prevalence := round(Smoking_Prevalence, 1)]
dt[, Drug_Experimentation := round(Drug_Experimentation, 1)]

clean_continuous_vars(
  dt, 
  cols = var_names[3:9], 
  min_val = 0, 
  max_val = 10, 
  round_digits = 0
)

# --- Variables conditionnelles -----------------------------------------------

# Age_Group conditionné à Smoking_Prevalence
age_def <- defCondition(condition = "Smoking_Prevalence < 25", dist = "categorical",
                        formula = "0.2;0.1;0.1;0.1;0.2;0.2;0.1")
age_def <- defCondition(age_def, condition = "Smoking_Prevalence >= 25",
                        dist = "categorical", formula = "0.1;0.3;0.25;0.15;0.1;0.05;0.05")
dt <- addCondition(age_def, dt, newvar = "Age_Group")

# Access_to_Counseling conditionné à Smoking_Prevalence
counseling_def <- defCondition(condition = "Smoking_Prevalence < 25", dist = "binary", formula = "0.05")
counseling_def <- defCondition(counseling_def, condition = "Smoking_Prevalence >= 25", dist = "binary", formula = "0.6")
dt <- addCondition(counseling_def, dt, newvar = "Access_to_Counseling")

# Socioeconomic_Status conditionné à Drug_Experimentation
socio_def <- defCondition(condition = "Drug_Experimentation < 10", dist = "categorical", formula = "0.4;0.2;0.4")
socio_def <- defCondition(socio_def, condition = "Drug_Experimentation >= 10", dist = "categorical", formula = "0.25;0.5;0.25")
dt <- addCondition(socio_def, dt, newvar = "Socioeconomic_Status")

# School_Programs conditionné à Drug_Experimentation
school_def <- defCondition(condition = "Drug_Experimentation < 10", dist = "categorical", formula = "0.2;0.8")
school_def <- defCondition(school_def, condition = "Drug_Experimentation >= 10", dist = "categorical", formula = "0.8;0.2")
dt <- addCondition(school_def, dt, newvar = "School_Programs")

# Autres variables simples
dt <- addColumns(defDataAdd(varname = "Gender", dist = "binary", formula = "0.5"), dt)

# --- Recodage -----------------------------------------------------------------

dt[, Socioeconomic_Status := factor(Socioeconomic_Status, labels = c("Low", "Medium", "High"))]
dt[, Gender := factor(Gender, labels = c("Female", "Male"))]
dt[, Access_to_Counseling := factor(Access_to_Counseling, labels = c("No", "Yes"))]
dt[, School_Programs := factor(School_Programs, labels = c("No", "Yes"))]
dt[, Age_Group := factor(Age_Group, labels = c("20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"))]

# --- Export CSV ---------------------------------------------------------------

fwrite(dt, "data/data_original.csv")

# --- Diagnostics -------------------------------------------------------------

print(table(dt$Age_Group, dt$Gender))
print(mean(dt[Gender == "Female", Drug_Experimentation]))
print(mean(dt[Gender == "Male", Drug_Experimentation]))

boxplot(Smoking_Prevalence ~ Age_Group, data = dt)
boxplot(Smoking_Prevalence ~ Access_to_Counseling, data = dt)

pairs(dt[, ..var_names])
hist(dt$Drug_Experimentation)
plot(dt$Drug_Experimentation, dt$Smoking_Prevalence)
print(cor(dt[, ..var_names]))

# ---------------------------------------------------------------------------------

# La version "sans corrélations" peut être mise en fonction pour éviter la redondance

