"""
color_kmeans.py
---------------
Parte de APRENDIZAJE NO SUPERVISADO del proyecto.

Obtiene el color dominante de una prenda agrupando sus píxeles con k-means.
El algoritmo k-means está implementado desde cero con NumPy para que se vea
cada paso (inicializar, asignar, actualizar, repetir).
"""

import colorsys

NEUTROS = {"negro", "blanco", "gris", "beige", "café", "azul marino"}
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. K-MEANS DESDE CERO
# ---------------------------------------------------------------------------
def kmeans(datos, k=4, max_iter=50, tol=1e-3, semilla=0):
    """
    Agrupa 'datos' (n x 3, valores RGB) en k grupos.

    Pasos:
      1) Elegir k centroides iniciales al azar (k-means++ simplificado).
      2) Asignar cada píxel al centroide más cercano (distancia euclidiana).
      3) Recalcular cada centroide como el promedio de sus píxeles.
      4) Repetir 2-3 hasta que los centroides casi no se muevan.

    Regresa: (centroides k x 3, etiquetas n)
    """
    rng = np.random.default_rng(semilla)
    datos = datos.astype(np.float64)
    n = len(datos)

    # Paso 1: inicialización k-means++ (el primer centroide al azar, los
    # siguientes con probabilidad proporcional a la distancia al más cercano)
    centroides = [datos[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((datos[:, None, :] - np.array(centroides)[None]) ** 2).sum(-1), axis=1)
        prob = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centroides.append(datos[rng.choice(n, p=prob)])
    centroides = np.array(centroides)

    for _ in range(max_iter):
        # Paso 2: asignación
        distancias = ((datos[:, None, :] - centroides[None]) ** 2).sum(-1)  # n x k
        etiquetas = distancias.argmin(axis=1)

        # Paso 3: actualización
        nuevos = np.array([
            datos[etiquetas == j].mean(axis=0) if np.any(etiquetas == j) else centroides[j]
            for j in range(k)
        ])

        # Paso 4: ¿convergió?
        movimiento = np.abs(nuevos - centroides).max()
        centroides = nuevos
        if movimiento < tol:
            break

    return centroides, etiquetas


# ---------------------------------------------------------------------------
# 2. PREPARAR LA IMAGEN
# ---------------------------------------------------------------------------
def cargar_pixeles(imagen, tamano=120, recorte=0.15):
    """
    Reduce la imagen (para que k-means sea rápido) y quita un margen del
    borde, porque ahí casi siempre está el fondo y no la prenda.
    Regresa (pixeles_centro n x 3, pixeles_borde m x 3).
    """
    if not isinstance(imagen, Image.Image):
        imagen = Image.open(imagen)
    imagen = imagen.convert("RGB")
    imagen.thumbnail((tamano, tamano))
    arr = np.asarray(imagen)
    h, w, _ = arr.shape
    dy, dx = int(h * recorte), int(w * recorte)

    centro = arr[dy:h - dy, dx:w - dx].reshape(-1, 3)
    borde = np.concatenate([
        arr[:dy].reshape(-1, 3), arr[h - dy:].reshape(-1, 3),
        arr[:, :dx].reshape(-1, 3), arr[:, w - dx:].reshape(-1, 3),
    ])
    return centro, borde


def color_dominante(imagen, k=4):
    """
    Devuelve el color dominante (R, G, B) de la prenda y la paleta completa.

    Truco para ignorar el fondo: se calcula el color promedio del borde de
    la foto; el grupo de k-means más parecido a ese color se descarta si
    es casi idéntico (se asume que es el fondo).
    """
    centro, borde = cargar_pixeles(imagen)
    centroides, etiquetas = kmeans(centro, k=k)
    tamanos = np.bincount(etiquetas, minlength=k)

    color_fondo = np.median(borde, axis=0)
    dist_fondo = np.linalg.norm(centroides - color_fondo, axis=1)

    orden = np.argsort(-tamanos)  # de grupo más grande a más chico
    paleta = [(tuple(int(c) for c in centroides[j]), tamanos[j] / tamanos.sum()) for j in orden]

    for j in orden:
        if dist_fondo[j] > 40:  # no se parece al fondo -> es la prenda
            return tuple(int(c) for c in centroides[j]), paleta
    # Si todo se parece al fondo (prenda del mismo color que el fondo)
    j = orden[0]
    return tuple(int(c) for c in centroides[j]), paleta


# ---------------------------------------------------------------------------
# 3. DESCRIBIR EL COLOR (para aplicar las reglas)
# ---------------------------------------------------------------------------
NOMBRES_TONO = [  # (grado máximo, nombre)
    (15, "rojo"), (40, "naranja"), (65, "amarillo"), (160, "verde"),
    (195, "turquesa"), (255, "azul"), (290, "morado"), (335, "rosa"), (360, "rojo"),
]


def describir_color(rgb):
    """
    Convierte RGB a HSV y clasifica el color.
    Regresa un diccionario con:
      nombre, tono (0-360), saturacion (0-1), brillo (0-1),
      neutro (bool), fuerte (bool)
    """
    r, g, b = [c / 255 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    tono = h * 360

    nombre = None
    if v < 0.22:
        nombre = "negro"
    elif s < 0.15 and v > 0.85:
        nombre = "blanco"
    elif s < 0.15:
        nombre = "gris"
    elif 15 <= tono <= 50 and s < 0.45 and v > 0.6:
        nombre = "beige"          # café muy claro y poco saturado
    elif 5 <= tono <= 50 and v < 0.65:
        nombre = "café"           # naranja/amarillo OSCURO = café o marrón
    elif 195 <= tono <= 260 and v < 0.5:
        nombre = "azul marino"    # azul oscuro

    if nombre is None:
        nombre = next(n for limite, n in NOMBRES_TONO if tono <= limite)
        # El rosa y el vino tienen el MISMO tono que el rojo; se distinguen
        # por la saturación (qué tan intenso) y el brillo (qué tan claro).
        if nombre in ("rojo", "rosa"):
            if v > 0.55 and s < 0.6:
                nombre = "rosa"      # rojo claro y deslavado (rosa palo, rosa pastel)
            elif v < 0.5:
                nombre = "vino"      # rojo oscuro (vino, guinda)
        elif nombre == "azul" and v > 0.75 and s < 0.5:
            nombre = "celeste"       # azul claro

    neutro = nombre in NEUTROS
    fuerte = (not neutro) and s > 0.6 and v > 0.5
    return {
        "nombre": nombre, "tono": tono, "saturacion": s, "brillo": v,
        "neutro": neutro, "fuerte": fuerte,
    }


def rgb_a_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)
