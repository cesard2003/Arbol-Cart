# 📌 Clasificación de Correos SPAM/HAM con Árbol de Decisión (CART)

Este proyecto corresponde a la **Actividad Semana 5** de la asignatura *Machine Learning*.  
El objetivo es implementar un **sistema clasificador de correos electrónicos (SPAM/HAM)** utilizando un **Árbol de Decisión (CART)** con la librería **scikit-learn** y el dataset previamente construido.

---

## 📂 Contenido del repositorio

- `Arbol Cart.py` → Script principal con la implementación del modelo CART.  
- `dataset_correos_1000_instancias.csv` → Dataset con 1000 correos (features + etiqueta HAM/SPAM).  
- Carpeta `spam_results/` → Se genera automáticamente al ejecutar el programa. Contiene:
  - `decision_tree_results.csv` → Resultados de 50 ejecuciones.  
  - `decision_tree_metrics.png` → Gráfica de métricas (Accuracy, F1, Z-score).  
  - `confusion_matrix_avg.png` → Matriz de confusión promedio.  
  - `decision_tree_feature_importance_avg.png` → Importancia de características (promedio de 50 ejecuciones).  
  - `decision_tree_structure.png` → Visualización del árbol CART.  
  - `metrics_distribution.png` → Distribución (boxplot) de métricas.  
  - `parameters_vs_metrics.png` → Relación entre parámetros y métricas.

---

## ⚙️ Requisitos

Instalar las librerías necesarias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn


