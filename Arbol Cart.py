# decision_tree_spam_mejorado.py
# Clasificación SPAM/HAM usando Árbol de Decisión (CART)
# Autor: [Cesar Aguirre]
# Semana 5 - Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.stats import zscore
from pathlib import Path

# ==============================
# BLOQUE 1: Preparación de datos
# ==============================
print("Cargando dataset...")
df = pd.read_csv(r"C:\Users\aguir\OneDrive\Escritorio\Machine Learning\dataset_correos_1000_instancias.csv")

# Features y target
X = df.drop("Label", axis=1)
y = df["Label"].map({"Ham": 0, "Spam": 1})

# Preprocesamiento
categorical_cols = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)]
)

# ==============================
# Crear carpeta de resultados
# ==============================
base_path = Path(__file__).resolve().parent
out_dir = base_path / "spam_results"
out_dir.mkdir(parents=True, exist_ok=True)

# ==============================
# BLOQUE 2-4: Modelo + Métricas en 50 ejecuciones
# ==============================
results = []
conf_matrices = []
feature_importances = []

# Obtener nombres de features codificados (una sola vez)
feature_names = preprocessor.fit(X).get_feature_names_out(categorical_cols)

for seed in range(1, 51):
    # Separar train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    # Crear pipeline con árbol de decisión (CART)
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            criterion="gini", random_state=seed,
            max_depth=None,
            min_samples_split=2
        ))
    ])

    # Entrenamiento
    clf.fit(X_train, y_train)

    # Predicciones
    y_pred = clf.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    results.append((seed, acc, f1))
    conf_matrices.append(cm)

    # Guardar importancia de features por ejecución (alineadas al total de columnas)
    importances = clf.named_steps["classifier"].feature_importances_
    if len(importances) < len(feature_names):
        padded = np.zeros(len(feature_names))
        padded[:len(importances)] = importances
        feature_importances.append(padded)
    else:
        feature_importances.append(importances)

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results, columns=["Seed", "Accuracy", "F1"])

# ==============================
# BLOQUE 5: Calcular Z-score
# ==============================
results_df["Z_Accuracy"] = zscore(results_df["Accuracy"])
results_df["Z_F1"] = zscore(results_df["F1"])

# Guardar resultados en CSV
results_df.to_csv(out_dir / "decision_tree_results.csv", index=False)
print(f"Resultados guardados en {out_dir/'decision_tree_results.csv'}")

# ==============================
# Graficar métricas (50 ejecuciones)
# ==============================
plt.figure(figsize=(12, 6))
plt.plot(results_df["Seed"], results_df["Accuracy"], label="Accuracy", marker="o")
plt.plot(results_df["Seed"], results_df["F1"], label="F1-score", marker="s")
plt.plot(results_df["Seed"], results_df["Z_Accuracy"], label="Z-Accuracy", linestyle="--")
plt.plot(results_df["Seed"], results_df["Z_F1"], label="Z-F1", linestyle="--")
plt.legend()
plt.xlabel("Iteración (seed)")
plt.ylabel("Métrica")
plt.title("Resultados Árbol de Decisión (CART) - 50 ejecuciones")
plt.tight_layout()
plt.savefig(out_dir / "decision_tree_metrics.png")
plt.show()

# ==============================
# BLOQUE 6: Reporte de clasificación en una corrida ejemplo
# ==============================
print("\nEjemplo de reporte de clasificación (última corrida):")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ==============================
# BLOQUE 7: Matriz de confusión promedio
# ==============================
avg_cm = np.mean(conf_matrices, axis=0)

plt.figure(figsize=(6, 5))
sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Matriz de confusión promedio (50 ejecuciones)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig(out_dir / "confusion_matrix_avg.png")
plt.show()

# ==============================
# BLOQUE 8: Importancia de características (promedio de 50 ejecuciones)
# ==============================
print("\nCalculando importancia promedio de características...")

feature_importances = np.array(feature_importances)
avg_importance = np.mean(feature_importances, axis=0)

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": avg_importance
}).sort_values(by="Importance", ascending=False)

feat_imp.to_csv(out_dir / "decision_tree_feature_importance_avg.csv", index=False)

top_n = 15
plt.figure(figsize=(10, 6))
bars = plt.barh(feat_imp["Feature"].head(top_n),
                feat_imp["Importance"].head(top_n),
                color=plt.cm.summer(np.linspace(0, 1, top_n)))

for bar, value in zip(bars, feat_imp["Importance"].head(top_n)):
    plt.text(value + 0.005, bar.get_y() + bar.get_height()/2,
             f"{value:.3f}", va="center", fontsize=9)

plt.xlabel("Importancia Promedio")
plt.title(f"Importancia Relativa de las Características\n(Promedio de 50 ejecuciones - Top {top_n})",
          fontsize=12, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(out_dir / "decision_tree_feature_importance_avg.png")
plt.show()

# ==============================
# BLOQUE 9: Visualización del árbol
# ==============================
plt.figure(figsize=(20, 10))
plot_tree(clf.named_steps["classifier"], filled=True, fontsize=6)
plt.title("Árbol de decisión CART")
plt.savefig(out_dir / "decision_tree_structure.png")
plt.show()

# ==============================
# BLOQUE 10: Gráficas adicionales
# ==============================

# Gráfica 2: Distribución de métricas (Boxplot)
plt.figure(figsize=(8, 6))
box_plot = plt.boxplot([results_df["Accuracy"], results_df["F1"]],
                       labels=["Accuracy", "F1 Score"], patch_artist=True)
colors = ["lightblue", "lightcoral"]
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
plt.title("Distribución de métricas (50 ejecuciones)", fontsize=14, fontweight='bold')
plt.ylabel("Valor de la Métrica")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "metrics_distribution.png")
plt.show()

# Gráfica 5: Relación entre parámetros y métricas
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(results_df["Seed"], results_df["Accuracy"], alpha=0.6, c="blue")
plt.xlabel("Seed")
plt.ylabel("Accuracy")
plt.title("Seed vs Accuracy")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(results_df["Seed"], results_df["F1"], alpha=0.6, c="red")
plt.xlabel("Seed")
plt.ylabel("F1 Score")
plt.title("Seed vs F1 Score")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(results_df["Accuracy"], results_df["F1"], alpha=0.6, c="green")
plt.xlabel("Accuracy")
plt.ylabel("F1 Score")
plt.title("Accuracy vs F1 Score")
plt.grid(True, alpha=0.3)

plt.suptitle("Relación entre parámetros y métricas", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(out_dir / "parameters_vs_metrics.png")
plt.show()

# ==============================
# Conclusiones rápidas
# ==============================
print("\nResumen general:")
print("Accuracy promedio:", results_df['Accuracy'].mean())
print("F1 promedio:", results_df['F1'].mean())
print("Desviación estándar Accuracy:", results_df['Accuracy'].std())
print("Desviación estándar F1:", results_df['F1'].std())
print("\nTop 5 características más importantes (promedio de 50 ejecuciones):")
print(feat_imp.head())

