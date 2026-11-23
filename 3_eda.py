"""
PASO 3 · ANÁLISIS EXPLORATORIO DE DATOS (EDA)
---------------------------------------------
Entrada : datos/dataset_limpio.csv
Salida  : gráficas en reportes/eda/  y  reportes/eda/resumen_eda.txt

  · Univariado   : cuántas fotos hay de cada tipo, de cada color, distribución del brillo.
  · Bivariado    : tipo de prenda vs color; brillo por tipo.
  · Multivariado : correlación entre características y PCA en 2D.
  · Nulos        : conteo de valores nulos.

Ejecutar:
    py -3.13 3_eda.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.caracteristicas import COLUMNAS_MODELO
from src.config import CLASES, COLORES, DATASET, EDA, NOMBRE_OCASION, OCASIONES, PATRONES

EDA.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATASET)
resumen = []
COLORES_CLASE = {"superior": "#4C72B0", "inferior": "#DD8452", "completo": "#C44E52", "calzado": "#55A868", "otro": "#8172B3"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(EDA / nombre)
    plt.close()
    print("  guardada:", EDA / nombre)


# ---------------- Nulos ----------------
nulos = int(df.isna().sum().sum())
resumen.append(f"Registros: {len(df)} · Columnas: {df.shape[1]} · Valores nulos: {nulos}")

# ---------------- Univariado ----------------
conteo = df["tipo"].value_counts().reindex(CLASES, fill_value=0)
plt.figure(figsize=(6, 4))
plt.bar(conteo.index, conteo.values, color=[COLORES_CLASE[c] for c in conteo.index])
plt.title("Univariado · Fotos por tipo de prenda")
plt.ylabel("Número de fotos")
guardar("01_univariado_tipo.png")
resumen.append("Fotos por tipo:\n" + conteo.to_string())

col = df["color"].value_counts()
plt.figure(figsize=(7, 4))
plt.bar(col.index, col.values, color="#888")
plt.xticks(rotation=45, ha="right")
plt.title("Univariado · Color real (etiquetado a mano)")
plt.ylabel("Número de fotos")
guardar("02_univariado_color.png")
resumen.append("Colores reales más frecuentes:\n" + col.head(8).to_string())

plt.figure(figsize=(6, 4))
plt.hist(df["dom_brillo"], bins=20, color="#4C72B0", edgecolor="white")
plt.title("Univariado · Brillo del color dominante")
plt.xlabel("Brillo (0 = negro, 1 = muy claro)")
guardar("03_univariado_brillo.png")

# ---------------- Bivariado ----------------
tabla = pd.crosstab(df["color"], df["tipo"]).reindex(columns=CLASES, fill_value=0)
tabla.plot(kind="bar", stacked=True, figsize=(8, 4.5),
           color=[COLORES_CLASE[c] for c in tabla.columns])
plt.title("Bivariado · Color real por tipo de prenda")
plt.ylabel("Número de fotos")
plt.xticks(rotation=45, ha="right")
guardar("04_bivariado_color_vs_tipo.png")

presentes = [c for c in CLASES if (df["tipo"] == c).any()]
plt.figure(figsize=(6, 4))
plt.boxplot([df.loc[df["tipo"] == c, "proporcion"] for c in presentes], tick_labels=presentes)
plt.title("Bivariado · Proporción alto/ancho de la foto por tipo")
plt.ylabel("alto / ancho")
guardar("05_bivariado_proporcion_vs_tipo.png")

# ---------------- Multivariado ----------------
principales = ["dom_r", "dom_g", "dom_b", "dom_saturacion", "dom_brillo",
               "es_neutro", "proporcion", "densidad_bordes"]
corr = df[principales].corr()
plt.figure(figsize=(6.5, 5.5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="correlación")
plt.xticks(range(len(principales)), principales, rotation=45, ha="right")
plt.yticks(range(len(principales)), principales)
for i in range(len(principales)):
    for j in range(len(principales)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Multivariado · Matriz de correlación")
guardar("06_multivariado_correlacion.png")

X = StandardScaler().fit_transform(df[COLUMNAS_MODELO])
pca = PCA(n_components=2, random_state=0)
puntos = pca.fit_transform(X)
plt.figure(figsize=(6.5, 5))
for c in presentes:
    m = (df["tipo"] == c).to_numpy()
    plt.scatter(puntos[m, 0], puntos[m, 1], s=14, alpha=.7, label=c, color=COLORES_CLASE[c])
var = pca.explained_variance_ratio_ * 100
plt.xlabel(f"Componente 1 ({var[0]:.1f}% de la varianza)")
plt.ylabel(f"Componente 2 ({var[1]:.1f}%)")
plt.title(f"Multivariado · PCA de las {len(COLUMNAS_MODELO)} características")
plt.legend()
guardar("07_multivariado_pca.png")
resumen.append(f"PCA: las 2 primeras componentes explican {var.sum():.1f}% de la varianza")

# ---------------- Nuevas variables: ocasiones, patrón y color detectado vs real ----------------
oc = pd.Series({NOMBRE_OCASION[o]: int(df[f"oc_{o}"].sum()) for o in OCASIONES})
plt.figure(figsize=(7, 4))
plt.bar(oc.index, oc.values, color="#4C72B0")
plt.xticks(rotation=20, ha="right")
plt.title("Univariado · Prendas por ocasión (una prenda puede tener varias)")
plt.ylabel("Número de prendas")
guardar("08_univariado_ocasiones.png")
resumen.append("Prendas por ocasión:\n" + oc.to_string())

pat = df["patron"].value_counts().reindex(PATRONES, fill_value=0)
plt.figure(figsize=(7, 4))
plt.bar(pat.index, pat.values, color="#DD8452")
plt.xticks(rotation=30, ha="right")
plt.title("Univariado · Patrón de las prendas")
plt.ylabel("Número de prendas")
guardar("09_univariado_patron.png")

presentes_c = [c for c in COLORES if c in set(df["color"]) | set(df["color_nombre"])]
cruce = pd.crosstab(df["color"], df["color_nombre"]).reindex(index=presentes_c, columns=presentes_c, fill_value=0)
aciertos = float((df["color"] == df["color_nombre"]).mean())
plt.figure(figsize=(8, 7))
plt.imshow(cruce, cmap="Blues")
plt.xticks(range(len(presentes_c)), presentes_c, rotation=45, ha="right")
plt.yticks(range(len(presentes_c)), presentes_c)
for i in range(len(presentes_c)):
    for j in range(len(presentes_c)):
        v = cruce.iloc[i, j]
        if v:
            plt.text(j, i, v, ha="center", va="center", fontsize=7)
plt.xlabel("Color detectado (k-means + reglas HSV)")
plt.ylabel("Color real (etiqueta)")
plt.title(f"Bivariado · Color real vs detectado: las reglas aciertan {100 * aciertos:.0f} %")
guardar("10_bivariado_color_real_vs_detectado.png")
resumen.append(f"Las reglas de color (k-means + HSV) aciertan el color real en {100 * aciertos:.1f} % de las prendas")

(EDA / "resumen_eda.txt").write_text("\n\n".join(resumen), encoding="utf-8")
print("\n" + "\n\n".join(resumen))
