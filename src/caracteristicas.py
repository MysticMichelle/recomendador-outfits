"""
caracteristicas.py
------------------
Convierte una foto en un vector de números (características) que los
modelos de aprendizaje automático pueden usar.

Características que se extraen:
  1. Color dominante (con k-means propio)       -> 9 columnas
  2. Histograma de color HSV (tono, saturación, brillo) -> 20 columnas
  3. Forma: histograma de orientaciones del gradiente (tipo HOG) -> 36 columnas
  4. Proporción alto/ancho y densidad de bordes  -> 2 columnas
"""

import numpy as np
from PIL import Image, ImageOps

from src.color_kmeans import color_dominante, describir_color

BINS_H, BINS_S, BINS_V = 12, 4, 4
CELDAS, BINS_ORIENT = 2, 9

COLUMNAS_MODELO = (
    ["dom_r", "dom_g", "dom_b", "dom_tono_sin", "dom_tono_cos", "dom_saturacion",
     "dom_brillo", "es_neutro", "es_fuerte"]
    + [f"hist_h{i}" for i in range(BINS_H)]
    + [f"hist_s{i}" for i in range(BINS_S)]
    + [f"hist_v{i}" for i in range(BINS_V)]
    + [f"forma_{i}" for i in range(CELDAS * CELDAS * BINS_ORIENT)]
    + ["proporcion", "densidad_bordes"]
)


def abrir(ruta_o_imagen):
    img = ruta_o_imagen if isinstance(ruta_o_imagen, Image.Image) else Image.open(ruta_o_imagen)
    img = ImageOps.exif_transpose(img)  # respeta la rotación de las fotos de celular
    return img.convert("RGB")


def _histograma_hsv(img):
    """Histogramas normalizados de tono, saturación y brillo del centro de la foto."""
    w, h = img.size
    centro = img.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))
    hsv = np.asarray(centro.convert("HSV")).reshape(-1, 3).astype(float)
    hh = np.histogram(hsv[:, 0], bins=BINS_H, range=(0, 256))[0]
    ss = np.histogram(hsv[:, 1], bins=BINS_S, range=(0, 256))[0]
    vv = np.histogram(hsv[:, 2], bins=BINS_V, range=(0, 256))[0]
    n = len(hsv)
    return np.concatenate([hh, ss, vv]) / n


def _forma(img):
    """
    Descriptor de forma simplificado (idea de HOG):
    se calcula el gradiente de la imagen en escala de grises y, en una
    cuadrícula de 2x2, un histograma de la dirección de los bordes.
    Un pantalón (bordes verticales) y una playera (hombros horizontales)
    producen histogramas distintos.
    """
    gris = np.asarray(img.convert("L").resize((64, 64)), dtype=float) / 255
    gy, gx = np.gradient(gris)
    magnitud = np.hypot(gx, gy)
    angulo = (np.degrees(np.arctan2(gy, gx)) + 180) % 180
    tam = 64 // CELDAS
    desc = []
    for i in range(CELDAS):
        for j in range(CELDAS):
            m = magnitud[i * tam:(i + 1) * tam, j * tam:(j + 1) * tam].ravel()
            a = angulo[i * tam:(i + 1) * tam, j * tam:(j + 1) * tam].ravel()
            hist = np.histogram(a, bins=BINS_ORIENT, range=(0, 180), weights=m)[0]
            desc.append(hist / (np.linalg.norm(hist) + 1e-6))
    densidad = float((magnitud > 0.1).mean())
    return np.concatenate(desc), densidad


def extraer(ruta_o_imagen):
    """
    Regresa un diccionario con todas las características de la foto,
    más 'color_nombre' (texto, para el EDA y las reglas de color).
    """
    img = abrir(ruta_o_imagen)
    proporcion = img.height / img.width
    img.thumbnail((256, 256))

    rgb, _ = color_dominante(img, k=4)
    c = describir_color(rgb)
    rad = np.radians(c["tono"])

    fila = {
        "dom_r": rgb[0], "dom_g": rgb[1], "dom_b": rgb[2],
        "dom_tono_sin": float(np.sin(rad)), "dom_tono_cos": float(np.cos(rad)),
        "dom_saturacion": c["saturacion"], "dom_brillo": c["brillo"],
        "es_neutro": int(c["neutro"]), "es_fuerte": int(c["fuerte"]),
    }
    for nombre, valor in zip(COLUMNAS_MODELO[9:9 + 20], _histograma_hsv(img)):
        fila[nombre] = float(valor)
    forma, densidad = _forma(img)
    for i, valor in enumerate(forma):
        fila[f"forma_{i}"] = float(valor)
    fila["proporcion"] = proporcion
    fila["densidad_bordes"] = densidad

    fila["color_nombre"] = c["nombre"]
    fila["dom_tono"] = c["tono"]
    return fila
