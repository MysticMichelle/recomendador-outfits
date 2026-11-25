"""
PASO 5 (EXPERIMENTO) · CNN PREENTRENADA (transfer learning)
------------------------------------------------------------
Pregunta: ¿mejora el reconocimiento si a cada foto se le agregan las características
aprendidas por una red neuronal convolucional preentrenada (MobileNetV2)?

No modifica nada de lo que ya funciona (modelos/modelos.joblib ni las métricas del
paso 4). Compara, con la MISMA partición 80/20 y la MISMA validación cruzada:

    · Manual        → las 67 características hechas a mano (las del paso 4)
    · CNN           → las 1000 salidas de MobileNetV2
    · CNN + manual  → las dos juntas (1067)

con los mismos 6 clasificadores. Si el mejor resultado con CNN supera en F1 de
validación cruzada al modelo actual, guarda modelos/modelos_cnn.joblib y la app lo
usa automáticamente. Si no lo supera, no guarda nada y todo sigue igual.

Requisitos (una sola vez):
    py -3.13 -m pip install onnxruntime
    Las fotos deben estar en datos/fotos/ (se leen otra vez para pasarlas por la red).

Ejecutar:
    py -3.13 5_cnn_experimento.py
"""

import matplotlib
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import cnn
from src.caracteristicas import COLUMNAS_MODELO
from src.config import (CLASES, DATASET, DATOS, FOTOS, METRICAS, METRICAS_OCASION, MODELADO, MODELOS,
                        NOMBRE_OCASION, OCASIONES, SEMILLA)

MODELO_CNN = MODELOS / "modelos_cnn.joblib"
CACHE = DATOS / "cnn_caracteristicas.csv"
MODELADO.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 0) Requisitos
# ---------------------------------------------------------------------------
try:
    import onnxruntime  # noqa: F401
except ImportError:
    raise SystemExit("Falta onnxruntime. Instálalo con:\n    py -3.13 -m pip install onnxruntime")

if not cnn.MODELO_ONNX.exists():
    print("Descargando MobileNetV2 (unos 14 MB)...")
    if not cnn.descargar():
        raise SystemExit(
            "No se pudo descargar el modelo. Descárgalo a mano desde:\n"
            f"    {cnn.URL_MODELO}\n"
            f"y guárdalo como:\n    {cnn.MODELO_ONNX}")
print(f"Modelo CNN: {cnn.MODELO_ONNX.name} ({cnn.MODELO_ONNX.stat().st_size / 1e6:.1f} MB)")

# ---------------------------------------------------------------------------
# 1) Características de la CNN (se guardan para no recalcularlas)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATASET)
previas = pd.read_csv(CACHE).set_index("archivo") if CACHE.exists() else pd.DataFrame()
faltan = [a for a in df["archivo"] if a not in previas.index]
if faltan:
    print(f"1) Pasando {len(faltan)} fotos por la CNN (puede tardar unos minutos)...")
    filas = []
    for i, archivo in enumerate(faltan, 1):
        ruta = FOTOS / archivo
        if not ruta.exists():
            raise SystemExit(f"No encuentro {ruta}. Las fotos deben estar en datos/fotos/.")
        filas.append([archivo, *cnn.vector(ruta)])
        if i % 50 == 0:
            print(f"   {i}/{len(faltan)}")
    nuevas = pd.DataFrame(filas, columns=["archivo", *cnn.COLUMNAS_CNN]).set_index("archivo")
    previas = pd.concat([previas, nuevas])
    previas.reset_index().to_csv(CACHE, index=False)
else:
    print("1) Características de la CNN ya calculadas (datos/cnn_caracteristicas.csv)")

C = previas.loc[df["archivo"], cnn.COLUMNAS_CNN].to_numpy()
M_ = df[COLUMNAS_MODELO].to_numpy()
y = df["tipo"].to_numpy()
conjuntos = {"Manual (67)": M_, "CNN (1000)": C, "CNN + manual (1067)": np.hstack([C, M_])}

# ---------------------------------------------------------------------------
# 2) Tipo de prenda: mismos 6 modelos, misma partición y misma validación cruzada
# ---------------------------------------------------------------------------
print("\n=== 2) Tipo de prenda: características manuales vs CNN ===")
modelos = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Regresión logística": LogisticRegression(max_iter=5000, class_weight="balanced"),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEMILLA),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEMILLA),
    "SVM (kernel RBF)": SVC(kernel="rbf", C=3, class_weight="balanced", random_state=SEMILLA),
    "Red neuronal (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=3000,
                                        random_state=SEMILLA),
}
indices = np.arange(len(y))
i_tr, i_te = train_test_split(indices, test_size=0.2, stratify=y, random_state=SEMILLA)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

filas = []
for nombre_conj, X in conjuntos.items():
    print(f"\n  -- {nombre_conj} --")
    for nombre, modelo in modelos.items():
        pipe = make_pipeline(StandardScaler(), clone(modelo))
        cv_scores = cross_val_score(pipe, X[i_tr], y[i_tr], cv=cv, scoring="f1_macro")
        pred = pipe.fit(X[i_tr], y[i_tr]).predict(X[i_te])
        filas.append({"caracteristicas": nombre_conj, "modelo": nombre,
                      "accuracy": accuracy_score(y[i_te], pred),
                      "f1": f1_score(y[i_te], pred, average="macro", zero_division=0),
                      "f1_cv_media": cv_scores.mean(), "f1_cv_desv": cv_scores.std()})
        r = filas[-1]
        print(f"  {nombre:22s} accuracy={r['accuracy']:.3f}  F1={r['f1']:.3f}  "
              f"F1 validación cruzada={r['f1_cv_media']:.3f}±{r['f1_cv_desv']:.3f}")

tabla = pd.DataFrame(filas).round(4)
tabla.to_csv(MODELADO / "experimento_cnn.csv", index=False)
mejores = tabla.loc[tabla.groupby("caracteristicas")["f1_cv_media"].idxmax()].set_index("caracteristicas")
print("\n  Mejor modelo con cada conjunto de características:")
print(mejores[["modelo", "accuracy", "f1", "f1_cv_media"]].to_string())

# Gráfica: F1 de validación cruzada por modelo y por conjunto de características
piv = tabla.pivot(index="modelo", columns="caracteristicas", values="f1_cv_media")[list(conjuntos)]
piv.plot(kind="barh", figsize=(8, 4.5), color=["#9aa5b8", "#b8892b", "#1f3a68"])
plt.xlabel("F1 macro (validación cruzada)")
plt.ylabel("")
plt.xlim(0, 1)
plt.title("Tipo de prenda: características hechas a mano vs CNN preentrenada")
plt.legend(title=None, loc="lower right")
plt.tight_layout()
plt.savefig(MODELADO / "04_experimento_cnn.png")
plt.close()

# ---------------------------------------------------------------------------
# 3) ¿Supera al modelo actual?
# ---------------------------------------------------------------------------
actual = pd.read_csv(METRICAS).iloc[0] if METRICAS.exists() else None
f1_actual = float(actual["f1_cv_media"]) if actual is not None else float(mejores.loc["Manual (67)", "f1_cv_media"])
candidatos = mejores.drop(index="Manual (67)")
mejor_conj = candidatos["f1_cv_media"].idxmax()
mejor = candidatos.loc[mejor_conj]
print(f"\n=== 3) Decisión ===\n  Modelo actual (paso 4): F1 validación cruzada = {f1_actual:.3f}")
print(f"  Mejor con CNN: {mejor['modelo']} con {mejor_conj}: F1 validación cruzada = {mejor['f1_cv_media']:.3f} "
      f"(accuracy en prueba = {mejor['accuracy']:.3f})")

if mejor["f1_cv_media"] <= f1_actual:
    if MODELO_CNN.exists():
        MODELO_CNN.unlink()
    print("  >>> La CNN NO mejora el score. Se conserva el modelo actual y no se guarda nada nuevo.")
    raise SystemExit(0)

print("  >>> La CNN SÍ mejora el score. Se guarda modelos/modelos_cnn.joblib y la app lo usará.")
X_mejor = conjuntos[mejor_conj]
modelo_nombre = mejor["modelo"]
base = modelos[modelo_nombre]
if isinstance(base, SVC):  # la app necesita el % de seguridad: se calibra para obtener probabilidades
    base = CalibratedClassifierCV(clone(base), cv=5)
final_tipo = make_pipeline(StandardScaler(), clone(base)).fit(X_mejor, y)

# ---------------------------------------------------------------------------
# 4) Ocasiones con las mismas características (umbral por ocasión, como en el paso 4)
# ---------------------------------------------------------------------------
print("\n=== 4) Ocasiones con características de la CNN ===")
ocas = [o for o in OCASIONES if f"oc_{o}" in df and df[f"oc_{o}"].sum() >= 5]
paquete_oc, filas_oc = None, []
if len(ocas) >= 2:
    Y = df[[f"oc_{o}" for o in ocas]].astype(int).to_numpy()
    onehot = np.column_stack([(y == c).astype(int) for c in CLASES])
    X_oc = np.hstack([X_mejor, onehot])
    candidatos_oc = {
        "Random Forest": MultiOutputClassifier(RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=SEMILLA)),
        "Regresión logística": MultiOutputClassifier(make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))),
    }
    cv_oc = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    mejor_fb, elegido = -1, None
    for nombre_oc, modelo_oc in candidatos_oc.items():
        P = np.zeros(Y.shape)
        for tr, te in cv_oc.split(X_oc):
            m = clone(modelo_oc).fit(X_oc[tr], Y[tr])
            P[te] = np.column_stack([p[:, 1] for p in m.predict_proba(X_oc[te])])
        umbrales, fbs = [], []
        for k in range(len(ocas)):
            fb = [fbeta_score(Y[:, k], (P[:, k] >= u).astype(int), beta=0.5, zero_division=0) for u in grid]
            j = int(np.argmax(fb))
            umbrales.append(float(grid[j]))
            fbs.append(fb[j])
        print(f"  {nombre_oc:20s} F0.5 promedio (validación cruzada, umbral ajustado) = {np.mean(fbs):.3f}")
        if np.mean(fbs) > mejor_fb:
            mejor_fb, elegido, P_mejor, umbrales_mejor = np.mean(fbs), nombre_oc, P, umbrales
    for k, o in enumerate(ocas):
        pred_k = (P_mejor[:, k] >= umbrales_mejor[k]).astype(int)
        fila = {"ocasion": NOMBRE_OCASION[o], "prendas": int(Y[:, k].sum()), "umbral": umbrales_mejor[k],
                "precision": round(precision_score(Y[:, k], pred_k, zero_division=0), 4),
                "recall": round(recall_score(Y[:, k], pred_k, zero_division=0), 4),
                "f1": round(f1_score(Y[:, k], pred_k, zero_division=0), 4), "modelo": elegido}
        filas_oc.append(fila)
        print(f"  {fila['ocasion']:18s} umbral={fila['umbral']:.2f}  precisión={fila['precision']:.3f}  "
              f"recall={fila['recall']:.3f}  F1={fila['f1']:.3f}")
    nuevo = pd.DataFrame(filas_oc)
    nuevo.to_csv(MODELADO / "experimento_cnn_ocasiones.csv", index=False)
    if METRICAS_OCASION.exists():
        viejo = pd.read_csv(METRICAS_OCASION)
        if "precision" in viejo:
            print(f"\n  Promedios  actual: precisión={viejo['precision'].mean():.3f} recall={viejo['recall'].mean():.3f} "
                  f"F1={viejo['f1'].mean():.3f}")
        print(f"  Promedios  CNN   : precisión={nuevo['precision'].mean():.3f} recall={nuevo['recall'].mean():.3f} "
              f"F1={nuevo['f1'].mean():.3f}")
    paquete_oc = {"modelo": clone(candidatos_oc[elegido]).fit(X_oc, Y), "ocasiones": ocas,
                  "nombre": elegido, "umbrales": dict(zip(ocas, umbrales_mejor))}

joblib.dump({"entrada": mejor_conj,
             "tipo": {"modelo": final_tipo, "nombre": f"{modelo_nombre} + CNN", "clases": list(final_tipo.classes_),
                      "con_probabilidad": hasattr(final_tipo, "predict_proba"),
                      "f1_cv": float(mejor["f1_cv_media"]), "accuracy": float(mejor["accuracy"])},
             "ocasion": paquete_oc},
            MODELO_CNN, compress=3)
print(f"\nGuardado en {MODELO_CNN} ({MODELO_CNN.stat().st_size / 1e6:.1f} MB)")
print("Gráfica: reportes/modelado/04_experimento_cnn.png")
