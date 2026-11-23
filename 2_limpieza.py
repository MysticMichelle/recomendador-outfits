"""
PASO 2 · LIMPIEZA DE DATOS
--------------------------
Entrada : datos/etiquetas/*.csv  +  datos/fotos/
Salida  : datos/dataset_limpio.csv   y   reportes/limpieza.txt

Qué hace:
  1. Une los CSV de etiquetas de todos los integrantes.
  2. HOMOLOGA el texto con expresiones regulares
     ("Playera", " PLAYERA", "arriba", "Superior"  ->  "superior";
      "marrón", "Cafe" -> "café"; "Business", "oficina" -> "business_casual").
  3. Quita filas sin etiqueta o marcadas como "descartar".
  4. Verifica que cada foto exista y no esté dañada.
  5. Quita fotos duplicadas (mismo contenido aunque tengan otro nombre).
  6. Extrae las características numéricas de cada foto.
  7. Revisa valores nulos y los IMPUTA (mediana, moda o color de k-means).

Ejecutar:
    py -3.13 2_limpieza.py
"""

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image

from src.caracteristicas import COLUMNAS_MODELO, extraer
from src.config import CLASES, COLORES, DATASET, ETIQUETAS, FOTOS, OCASIONES, PATRONES, REPORTES

REPORTES.mkdir(parents=True, exist_ok=True)
bitacora = []


def log(texto):
    print(texto)
    bitacora.append(texto)


# ---------------------------------------------------------------------------
# 1. Unir CSV
# ---------------------------------------------------------------------------
archivos = sorted(ETIQUETAS.glob("*.csv"))
if not archivos:
    raise SystemExit("No hay etiquetas. Primero corre: python -m streamlit run 1_etiquetar.py")

df = pd.concat([pd.read_csv(a, dtype=str) for a in archivos], ignore_index=True)
log(f"1) Registros leídos de {len(archivos)} archivo(s): {len(df)}")

# ---------------------------------------------------------------------------
# 2. Homologación con expresiones regulares
# ---------------------------------------------------------------------------
def normalizar(texto):
    """minúsculas, sin acentos, sin espacios repetidos."""
    if pd.isna(texto):
        return np.nan
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t or np.nan


REGLAS_TIPO = [
    (r"^(comp|cuerpo|vestid|overol|jumpsuit|enteriz|mono$)", "completo"),
    (r"^(sup|arriba|playera|camis|blusa|sueter|sudadera|chamarra|saco|top|polo|abrigo|chaleco)", "superior"),
    (r"^(inf|abajo|pantal|falda|short|jean|bermuda|mallas|leggin)", "inferior"),
    (r"^(calz|zapat|teni|bota|sandal|huarach|flat)", "calzado"),
    (r"^(otr|bolsa|bolso|accesor|gorra|cinturon|mochila)", "otro"),
    (r"^(desc|borr|basura|x$)", "descartar"),
]


def homologar_tipo(texto):
    t = normalizar(texto)
    if pd.isna(t):
        return np.nan
    for patron, clase in REGLAS_TIPO:
        if re.search(patron, t):
            return clase
    return np.nan  # texto que no se reconoce -> nulo


# --- color, patrón y ocasiones (mismo principio: normalizar y mapear sinónimos)
SINONIMOS_COLOR = {"marron": "cafe", "cafe": "cafe", "chocolate": "cafe", "camel": "cafe",
                   "marino": "azul marino", "azul marino": "azul marino", "navy": "azul marino",
                   "roho": "rojo", "blanca": "blanco", "negra": "negro", "guinda": "vino",
                   "celeste": "celeste", "azul claro": "celeste"}
COLORES_NORM = {normalizar(c): c for c in COLORES}


def homologar_color(texto):
    t = normalizar(texto)
    if pd.isna(t):
        return np.nan
    t = SINONIMOS_COLOR.get(t, t)
    return COLORES_NORM.get(normalizar(t), np.nan)


PATRONES_NORM = {normalizar(x): x for x in PATRONES}


def homologar_patron(texto):
    t = normalizar(texto)
    if pd.isna(t) or re.match(r"^(no|ninguno|sin|liso)", t):
        return "liso"
    if re.search(r"logo|grafic", t):
        return "gráfico / logo"
    if re.search(r"animal|leopard|cebra", t):
        return "animal print"
    return PATRONES_NORM.get(t, "otro estampado")


def homologar_ocasiones(texto):
    t = normalizar(texto)
    if pd.isna(t):
        return ""
    salida = []
    for parte in re.split(r"[|,;/]", t):
        parte = parte.strip().replace(" ", "_")
        if re.match(r"^(casual|diario)", parte):
            salida.append("casual")
        elif re.match(r"^(business|oficina|trabajo)", parte):
            salida.append("business_casual")
        elif re.match(r"^(formal|evento)", parte):
            salida.append("formal")
        elif re.match(r"^(deport|gym)", parte):
            salida.append("deportivo")
        elif re.match(r"^(fiesta|noche|antro)", parte):
            salida.append("fiesta")
    return "|".join(o for o in OCASIONES if o in salida)


for columna in ["color", "patron", "mezclilla", "ocasiones"]:
    if columna not in df.columns:
        df[columna] = np.nan
df["color"] = df["color"].map(homologar_color)
df["patron"] = df["patron"].map(homologar_patron)
df["mezclilla"] = df["mezclilla"].map(lambda x: 1 if re.match(r"^(1|si|s|true|x)$", str(normalizar(x))) else 0)
df["ocasiones"] = df["ocasiones"].map(homologar_ocasiones)
df["estampado"] = (df["patron"] != "liso").astype(int)
for o in OCASIONES:
    df[f"oc_{o}"] = df["ocasiones"].str.split("|").map(lambda lista, o=o: int(o in lista))

antes = df["tipo"].dropna().unique().tolist()
df["tipo"] = df["tipo"].map(homologar_tipo)
df["persona"] = df["persona"].map(lambda x: normalizar(x).title() if not pd.isna(normalizar(x)) else np.nan)
df["archivo"] = df["archivo"].str.strip().str.replace("\\", "/", regex=False)
log(f"2) Homologación: valores de 'tipo' originales {antes} -> {sorted(df['tipo'].dropna().unique())}")
log(f"   colores: {df['color'].value_counts().to_dict()}")
log(f"   patrones: {df['patron'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 3. Filas sin etiqueta / descartadas / repetidas en el CSV
# ---------------------------------------------------------------------------
n = len(df)
df = df[df["tipo"].isin(CLASES)]
df = df.drop_duplicates(subset="archivo", keep="last")
log(f"3) Quitadas {n - len(df)} filas (sin etiqueta válida, descartadas o repetidas)")

# ---------------------------------------------------------------------------
# 4. Fotos inexistentes o dañadas  +  5. duplicados por contenido
# ---------------------------------------------------------------------------
validas, hashes, faltan, danadas = [], [], 0, 0
for archivo in df["archivo"]:
    ruta = FOTOS / archivo
    if not ruta.exists():
        faltan += 1
        validas.append(False)
        hashes.append(None)
        continue
    try:
        with Image.open(ruta) as im:
            im.verify()
        validas.append(True)
        hashes.append(hashlib.md5(ruta.read_bytes()).hexdigest())
    except Exception:
        danadas += 1
        validas.append(False)
        hashes.append(None)
df = df[validas].copy()
df["hash"] = [h for h, v in zip(hashes, validas) if v]
log(f"4) Fotos que no existen: {faltan} · fotos dañadas: {danadas}")

n = len(df)
df = df.drop_duplicates(subset="hash")
log(f"5) Fotos duplicadas (mismo contenido) eliminadas: {n - len(df)}")

# ---------------------------------------------------------------------------
# 6. Extracción de características
# ---------------------------------------------------------------------------
log(f"6) Extrayendo características de {len(df)} fotos (puede tardar unos minutos)...")
filas = []
for i, archivo in enumerate(df["archivo"], 1):
    try:
        filas.append(extraer(FOTOS / archivo))
    except Exception as e:  # si una foto falla, sus características quedan nulas
        log(f"   ! No se pudo procesar {archivo}: {e}")
        filas.append({})
    if i % 50 == 0:
        print(f"   {i}/{len(df)}")
caract = pd.DataFrame(filas, index=df.index).reindex(columns=COLUMNAS_MODELO + ["color_nombre", "dom_tono"])
df = pd.concat([df, caract], axis=1)

# ---------------------------------------------------------------------------
# 7. Valores nulos e imputación
# ---------------------------------------------------------------------------
nulos = df[COLUMNAS_MODELO + ["color_nombre", "color"]].isna().sum()
log(f"7) Valores nulos por columna (solo las que tienen): {nulos[nulos > 0].to_dict() or 'ninguno'}")
for col in COLUMNAS_MODELO + ["dom_tono"]:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
if df["color_nombre"].isna().any():
    df["color_nombre"] = df["color_nombre"].fillna(df["color_nombre"].mode()[0])
sin_color = int(df["color"].isna().sum())
df["color"] = df["color"].fillna(df["color_nombre"])
log(f"   Imputación: numéricas -> mediana, color_nombre -> moda, color real vacío ({sin_color}) -> color de k-means")

df = df.drop(columns=["hash", "fecha"], errors="ignore")
DATASET.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DATASET, index=False, encoding="utf-8")
log(f"\nDataset limpio: {len(df)} fotos x {df.shape[1]} columnas -> {DATASET}")
log("Fotos por tipo:\n" + df["tipo"].value_counts().to_string())

(REPORTES / "limpieza.txt").write_text("\n".join(bitacora), encoding="utf-8")
