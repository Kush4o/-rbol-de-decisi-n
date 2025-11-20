import numpy as np
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print(f"Número total de muestras: {len(X)}")
print(f"Número de características: {X.shape[1]}")
print(f"Clases de vino: {target_names}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Muestras de entrenamiento: {len(X_train)}")
print(f"Muestras de prueba: {len(X_test)}\n")

tree_depth_2 = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_depth_2.fit(X_train, y_train)

print("Reglas del Árbol de Decisión (max_depth=4)")
rules_depth_2 = export_text(tree_depth_2, feature_names=feature_names)
print(rules_depth_2)

accuracy_depth_2 = tree_depth_2.score(X_test, y_test)
print(f"Precisión en datos de prueba (max_depth=4): {accuracy_depth_2:.4f}\n")

tree_full = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_full.fit(X_train, y_train)

print("Reglas del Árbol de Decisión (max_depth=None - Árbol completo)")
rules_full_snippet = export_text(tree_full, feature_names=feature_names, max_depth=4)
print(rules_full_snippet)
print("[... Árbol continúa con más reglas anidadas ...]\n")

accuracy_full = tree_full.score(X_test, y_test)
print(f"Precisión en datos de prueba (max_depth=None): {accuracy_full:.4f}")