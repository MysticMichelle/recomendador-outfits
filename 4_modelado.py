"""
PASO 4 · MODELADO
-----------------
Entrada : datos/dataset_limpio.csv
Salida  : reportes/modelado/ (gráficas y métricas)  y  modelos/clasificador.joblib

A) APRENDIZAJE NO SUPERVISADO
   · k-means (implementado desde cero en src/color_kmeans.py) ya se usó para
     obtener el color dominante de cada foto.
   · Aquí además se agrupan TODAS las fotos con k-means (scikit-learn) para ver
     si se forman grupos naturales. Se elige k con el método del codo y el
     coeficiente de silueta.

B) APRENDIZAJE SUPERVISADO
   · Objetivo: predecir el tipo de prenda (superior / inferior / calzado / otro).
   · Se entrenan 6 modelos y se comparan con accuracy, precisión, recall y F1,
     usando 80 % de las fotos para entrenar y 20 % para probar, más validación
     cruzada de 5 particiones.
   · Las clases están desbalanceadas, así que se usa class_weight="balanced".
   · El mejor modelo (mayor F1 en validación cruzada) se guarda y lo usa la app.

C) COLOR: se compara qué acierta más el nombre del color real:
   las reglas HSV sobre el color de k-means, o un Random Forest entrenado
   con los colores etiquetados a mano.

D) OCASIONES (multi-etiqueta): un Random Forest sugiere para qué ocasiones
   sirve una prenda (casual, business casual, formal, deportivo, fiesta).

Ejecutar:
    py -3.13 4_modelado.py
"""

import matplotlib
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, silhouette_score)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.caracteristicas import COLUMNAS_MODELO
from src.config import (CLASES, DATASET, METRICAS, METRICAS_COLOR, METRICAS_OCASION, MODELADO,
                        MODELO_FINAL, MODELOS, NOMBRE_OCASION, OCASIONES, SEMILLA)

MODELADO.mkdir(parents=True, exist_ok=True)
MODELOS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET)
X = df[COLUMNAS_MODELO].to_numpy()
y = df["tipo"].to_numpy()

minimo = df["tipo"].value_counts().min()
if minimo < 5:
    raise SystemExit(f"Cada tipo necesita al menos 5 fotos (el más chico tiene {minimo}). Etiqueten más fotos.")

# ===========================================================================
# A) NO SUPERVISADO · k-means sobre todas las fotos
# ===========================================================================
print("=== A) No supervisado: k-means ===")
Xs = StandardScaler().fit_transform(X)
ks = list(range(2, 9))
inercias, siluetas = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Xs)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(Xs, km.labels_))
    print(f"  k={k}: inercia={km.inertia_:.0f}  silueta={siluetas[-1]:.3f}")
mejor_k = ks[int(np.argmax(siluetas))]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inercias, "o-")
ax[0].set(title="Método del codo", xlabel="k (número de grupos)", ylabel="Inercia")
ax[1].plot(ks, siluetas, "o-", color="#55A868")
ax[1].axvline(mejor_k, ls="--", color="gray")
ax[1].set(title=f"Coeficiente de silueta (mejor k = {mejor_k})", xlabel="k", ylabel="Silueta")
plt.tight_layout()
plt.savefig(MODELADO / "01_kmeans_codo_silueta.png")
plt.close()

km = KMeans(n_clusters=mejor_k, n_init=10, random_state=SEMILLA).fit(Xs)
df["cluster"] = km.labels_
tabla = pd.crosstab(df["cluster"], df["tipo"])
tabla.to_csv(MODELADO / "kmeans_clusters_vs_tipo.csv")
print(f"\n  Grupos encontrados (k={mejor_k}) comparados con el tipo real:\n{tabla}\n")

# ===========================================================================
# B) SUPERVISADO · comparación de modelos
# ===========================================================================
print("=== B) Supervisado: clasificación del tipo de prenda ===")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
print(f"  Entrenamiento: {len(X_tr)} fotos · Prueba: {len(X_te)} fotos")

modelos = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Regresión logística": LogisticRegression(max_iter=5000, class_weight="balanced"),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEMILLA),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEMILLA),
    "SVM (kernel RBF)": SVC(kernel="rbf", C=3, class_weight="balanced", random_state=SEMILLA),
    "Red neuronal (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=3000,
                                        random_state=SEMILLA),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
resultados, entrenados = [], {}
for nombre, modelo in modelos.items():
    pipe = make_pipeline(StandardScaler(), modelo)
    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    resultados.append({
        "modelo": nombre,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, average="macro", zero_division=0),
        "recall": recall_score(y_te, pred, average="macro", zero_division=0),
        "f1": f1_score(y_te, pred, average="macro", zero_division=0),
        "f1_cv_media": cv_scores.mean(),
        "f1_cv_desv": cv_scores.std(),
    })
    entrenados[nombre] = pipe
    r = resultados[-1]
    print(f"  {nombre:22s} accuracy={r['accuracy']:.3f}  F1={r['f1']:.3f}  "
          f"F1 validación cruzada={r['f1_cv_media']:.3f}±{r['f1_cv_desv']:.3f}")

tabla = pd.DataFrame(resultados).sort_values(["f1_cv_media", "f1"], ascending=False).round(4)
tabla.to_csv(METRICAS, index=False)
mejor = tabla.iloc[0]["modelo"]
# Se elige por F1 de VALIDACIÓN CRUZADA (promedio de 5 pruebas), no por una sola
# partición de prueba, para no premiar a un modelo que tuvo "suerte".
print(f"\n  >>> Mejor modelo (mayor F1 en validación cruzada): {mejor}")

# Gráfica comparativa
plt.figure(figsize=(8, 4))
pos = np.arange(len(tabla))
plt.bar(pos - 0.2, tabla["accuracy"], 0.4, label="Accuracy")
plt.bar(pos + 0.2, tabla["f1"], 0.4, label="F1 (macro)")
plt.xticks(pos, tabla["modelo"], rotation=20, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Comparación de modelos supervisados (conjunto de prueba)")
plt.legend()
plt.tight_layout()
plt.savefig(MODELADO / "02_comparacion_modelos.png")
plt.close()

# Matriz de confusión y reporte del mejor modelo
clases = [c for c in CLASES if c in set(y)]
pred = entrenados[mejor].predict(X_te)
ConfusionMatrixDisplay.from_predictions(y_te, pred, labels=clases, cmap="Blues")
plt.title(f"Matriz de confusión · {mejor}")
plt.tight_layout()
plt.savefig(MODELADO / "03_matriz_confusion.png")
plt.close()
reporte = classification_report(y_te, pred, labels=clases, zero_division=0)
(MODELADO / "reporte_clasificacion.txt").write_text(f"Modelo: {mejor}\n\n{reporte}", encoding="utf-8")
print("\n" + reporte)

# El modelo final se reentrena con TODAS las fotos y se guarda para la app
final = entrenados[mejor]
final.fit(X, y)
paquete = {"tipo": {"modelo": final, "nombre": mejor, "clases": list(final.classes_),
                    "con_probabilidad": hasattr(final, "predict_proba")},
           "columnas": COLUMNAS_MODELO}

# ===========================================================================
# C) COLOR · reglas HSV vs modelo entrenado con el color real
# ===========================================================================
print("\n=== C) Color: reglas HSV vs Random Forest ===")
conteo = df["color"].value_counts()
validos = conteo[conteo >= 5].index           # colores con al menos 5 fotos
dc = df[df["color"].isin(validos)]
acc_reglas = float((dc["color"] == dc["color_nombre"]).mean())
rf_color = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                                  random_state=SEMILLA))
pred_c = cross_val_predict(rf_color, dc[COLUMNAS_MODELO].to_numpy(), dc["color"].to_numpy(),
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA))
acc_rf = float((pred_c == dc["color"].to_numpy()).mean())
print(f"  Fotos usadas: {len(dc)} · colores con >= 5 fotos: {len(validos)}")
print(f"  Reglas HSV (k-means)      accuracy = {acc_reglas:.3f}")
print(f"  Random Forest de color    accuracy = {acc_rf:.3f}  (validación cruzada)")
usar_modelo_color = acc_rf > acc_reglas
pd.DataFrame([{"metodo": "Reglas HSV sobre k-means", "accuracy": round(acc_reglas, 4)},
              {"metodo": "Random Forest de color", "accuracy": round(acc_rf, 4)}]).to_csv(METRICAS_COLOR, index=False)
print(f"  >>> La app usará: {'Random Forest de color' if usar_modelo_color else 'reglas HSV'}")
if usar_modelo_color:
    rf_color.fit(dc[COLUMNAS_MODELO].to_numpy(), dc["color"].to_numpy())
    paquete["color"] = {"modelo": rf_color, "accuracy": acc_rf}
else:
    paquete["color"] = None

# ===========================================================================
# D) OCASIONES · Random Forest multi-etiqueta
# ===========================================================================
print("\n=== D) Ocasiones (multi-etiqueta) ===")
tipos_onehot = pd.get_dummies(df["tipo"]).reindex(columns=CLASES, fill_value=0).astype(int)
X_oc = np.hstack([X, tipos_onehot.to_numpy()])
ocas = [o for o in OCASIONES if df[f"oc_{o}"].sum() >= 5 and (1 - df[f"oc_{o}"]).sum() >= 5]
filas_oc = []
if ocas:
    Y = df[[f"oc_{o}" for o in ocas]].to_numpy()
    # Un clasificador binario POR ocasión (MultiOutputClassifier). Un solo Random Forest
    # multi-salida con class_weight="balanced" multiplica los pesos de todas las ocasiones
    # y las prendas raras dominan (casi nunca predice "casual"), por eso se separan.
    candidatos_oc = {
        "Random Forest": MultiOutputClassifier(RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=SEMILLA)),
        "Regresión logística": MultiOutputClassifier(make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))),
    }
    cv_oc = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    mejor_oc, mejor_f1, mejor_pred = None, -1, None
    for nombre_oc, modelo_oc in candidatos_oc.items():
        pred_o = cross_val_predict(modelo_oc, X_oc, Y, cv=cv_oc)
        f1s = [f1_score(Y[:, k], pred_o[:, k], zero_division=0) for k in range(len(ocas))]
        print(f"  {nombre_oc:20s} F1 promedio (validación cruzada) = {np.mean(f1s):.3f}")
        if np.mean(f1s) > mejor_f1:
            mejor_oc, mejor_f1, mejor_pred = nombre_oc, np.mean(f1s), pred_o
    print(f"  >>> Modelo de ocasiones: {mejor_oc}")
    for k, o in enumerate(ocas):
        f1 = f1_score(Y[:, k], mejor_pred[:, k], zero_division=0)
        filas_oc.append({"ocasion": NOMBRE_OCASION[o], "prendas": int(Y[:, k].sum()), "f1": round(f1, 4), "modelo": mejor_oc})
        print(f"  {NOMBRE_OCASION[o]:18s} prendas={int(Y[:, k].sum()):4d}  F1={f1:.3f}")
    modelo_final_oc = candidatos_oc[mejor_oc].fit(X_oc, Y)
    paquete["ocasion"] = {"modelo": modelo_final_oc, "ocasiones": ocas, "nombre": mejor_oc}
else:
    print("  No hay suficientes prendas por ocasión para entrenar.")
    paquete["ocasion"] = None
pd.DataFrame(filas_oc).to_csv(METRICAS_OCASION, index=False)

# compress=3 reduce mucho el tamaño (el Random Forest de color ocupa >25 MB sin comprimir)
joblib.dump(paquete, MODELO_FINAL, compress=3)
print(f"Tamaño del archivo de modelos: {MODELO_FINAL.stat().st_size / 1e6:.1f} MB")
print(f"\nModelos guardados en {MODELO_FINAL}")
