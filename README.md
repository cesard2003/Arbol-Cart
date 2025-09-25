🌐 Clasificador SPAM/HAM con Árbol de Decisión (CART)

Autores: Cesar Aguirre [y otro integrante si aplica]
Curso: Machine Learning – Semana 5




📌 Descripción del proyecto

Este proyecto implementa un clasificador de correos electrónicos para distinguir entre SPAM y HAM utilizando un Árbol de Decisión (CART) con la librería scikit-learn.

El objetivo principal es:

Entrenar un modelo que clasifique correos electrónicos correctamente.
Evaluar su desempeño mediante Accuracy, F1 Score y Z-score.
Repetir la ejecución 50 veces con diferentes particiones de entrenamiento y prueba para analizar estabilidad y robustez.
Identificar las características más importantes que afectan la clasificación.ç

🗂 Estructura del proyecto
Arbol-Cart/
│
├─ decision_tree_spam_mejorado.py       # Código principal
├─ spam_results/                        # Carpeta con resultados y gráficos
│   ├─ decision_tree_metrics.png
│   ├─ confusion_matrix_avg.png
│   ├─ decision_tree_feature_importance_avg.png
│   ├─ decision_tree_structure.png
│   ├─ metrics_distribution.png
│   └─ decision_tree_results.csv
├─ README.md                            # Este archivo
└─ dataset_correos_1000_instancias.csv  # Dataset de correos

