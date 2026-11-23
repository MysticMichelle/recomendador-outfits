"""Rutas, clases y listas de valores compartidas por todo el proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DATOS = RAIZ / "datos"
FOTOS = DATOS / "fotos"                    # datos/fotos/<Persona>/<foto>.jpg
ETIQUETAS = DATOS / "etiquetas"            # un CSV por persona (lo genera 1_etiquetar.py)
DATASET = DATOS / "dataset_limpio.csv"     # salida de 2_limpieza.py

REPORTES = RAIZ / "reportes"
EDA = REPORTES / "eda"
MODELADO = REPORTES / "modelado"
MODELOS = RAIZ / "modelos"
MODELO_FINAL = MODELOS / "modelos.joblib"
METRICAS = MODELADO / "metricas_modelos.csv"
METRICAS_COLOR = MODELADO / "metricas_color.csv"
METRICAS_OCASION = MODELADO / "metricas_ocasion.csv"

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}
SEMILLA = 42

# ---------------------------------------------------------------- tipo de prenda
CLASES = ["superior", "inferior", "completo", "calzado", "otro"]
NOMBRE_CLASE = {
    "superior": "👕 Arriba",
    "inferior": "👖 Abajo",
    "completo": "👗 Completo",
    "calzado": "👟 Calzado",
    "otro": "👜 Otro",
}
DESCRIPCION_CLASE = {
    "superior": "playera, blusa, camisa, top, suéter, sudadera, chamarra, saco",
    "inferior": "pantalón, jeans, falda, short",
    "completo": "vestido, overol, jumpsuit, enterizo",
    "calzado": "tenis, zapatos, botas, sandalias",
    "otro": "bolsa, gorra, cinturón, accesorios",
}

# ---------------------------------------------------------------- color real
COLORES = ["negro", "blanco", "gris", "beige", "café", "azul marino", "azul", "celeste",
           "turquesa", "verde", "amarillo", "naranja", "rojo", "vino", "rosa", "morado", "multicolor"]
NEUTROS = {"negro", "blanco", "gris", "beige", "café", "azul marino"}
# Tono aproximado (grados) de cada color, para las reglas de la rueda de color
TONO_COLOR = {"azul": 225, "celeste": 205, "turquesa": 178, "verde": 115, "amarillo": 52,
              "naranja": 28, "rojo": 2, "vino": 350, "rosa": 330, "morado": 275,
              "negro": 0, "blanco": 0, "gris": 0, "beige": 35, "café": 30, "azul marino": 225, "multicolor": 0}

# ---------------------------------------------------------------- patrón
PATRONES = ["liso", "rayas", "cuadros", "floral", "lunares", "animal print", "gráfico / logo", "otro estampado"]

# ---------------------------------------------------------------- ocasiones
OCASIONES = ["casual", "business_casual", "formal", "deportivo", "fiesta"]
NOMBRE_OCASION = {
    "casual": "Casual / diario",
    "business_casual": "Business casual",
    "formal": "Formal / evento",
    "deportivo": "Deportivo",
    "fiesta": "Fiesta / noche",
}

UMBRAL_SEGURIDAD = 0.60   # debajo de esto la app avisa que no está segura

# ---------------------------------------------------------------- Mi clóset
CLOSETS = DATOS / "closets"      # datos/closets/<usuario>/
DIAS_GUARDADO = 30               # días que se conserva cada prenda guardada
ALMACEN = "local"                # "local" hoy; "nube" cuando se implemente AlmacenNube
ESTILO_CSS = RAIZ / "assets" / "estilo.css"   # estilos visuales de la app


def listar_fotos():
    """Regresa la lista de rutas relativas 'Persona/foto.jpg' dentro de datos/fotos."""
    if not FOTOS.exists():
        return []
    return sorted(
        p.relative_to(FOTOS).as_posix()
        for p in FOTOS.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONES
    )
