"""
Características aprendidas con una red neuronal convolucional (CNN) preentrenada.
--------------------------------------------------------------------------------
Se usa MobileNetV2, una CNN entrenada con más de un millón de fotos (ImageNet).
No se vuelve a entrenar la red: cada foto pasa por ella y se toman sus 1000 salidas
como un vector de características ("transfer learning" por extracción de
características). Con esos vectores se entrenan los mismos clasificadores de
scikit-learn que con las características hechas a mano.

Se usa el formato ONNX con la librería onnxruntime (unos 15 MB) en lugar de
PyTorch o TensorFlow (cientos de MB), para que el proyecto se pueda instalar en
cualquier computadora y publicar en Streamlit Community Cloud.

El archivo del modelo (unos 14 MB) se guarda en modelos/mobilenetv2-12.onnx.
"""

import urllib.request

import numpy as np
from PIL import Image, ImageOps

from src.config import MODELOS

MODELO_ONNX = MODELOS / "mobilenetv2-12.onnx"
URL_MODELO = ("https://github.com/onnx/models/raw/main/validated/vision/"
              "classification/mobilenet/model/mobilenetv2-12.onnx")
DIMENSION = 1000

# Normalización con la que se entrenó la red en ImageNet
_MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DESV = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_sesion = None


def disponible():
    """True si están instalados onnxruntime y el archivo del modelo."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return MODELO_ONNX.exists()


def descargar():
    """Descarga el modelo MobileNetV2 (formato ONNX) si todavía no está."""
    if MODELO_ONNX.exists():
        return True
    MODELOS.mkdir(parents=True, exist_ok=True)
    temporal = MODELO_ONNX.with_suffix(".descargando")
    try:
        urllib.request.urlretrieve(URL_MODELO, temporal)
        if temporal.stat().st_size < 1_000_000:  # se descargó una página de error, no el modelo
            temporal.unlink()
            return False
        temporal.replace(MODELO_ONNX)
        return True
    except Exception:
        if temporal.exists():
            temporal.unlink()
        return False


def _obtener_sesion():
    global _sesion
    if _sesion is None:
        import onnxruntime as ort
        _sesion = ort.InferenceSession(str(MODELO_ONNX), providers=["CPUExecutionProvider"])
    return _sesion


def _preparar(ruta_o_imagen):
    """Misma preparación que en ImageNet: lado corto a 256, recorte central de 224, normalizar."""
    img = ruta_o_imagen if isinstance(ruta_o_imagen, Image.Image) else Image.open(ruta_o_imagen)
    img = ImageOps.exif_transpose(img).convert("RGB")
    ancho, alto = img.size
    escala = 256 / min(ancho, alto)
    img = img.resize((max(224, round(ancho * escala)), max(224, round(alto * escala))), Image.BILINEAR)
    ancho, alto = img.size
    izq, arr = (ancho - 224) // 2, (alto - 224) // 2
    img = img.crop((izq, arr, izq + 224, arr + 224))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - _MEDIA) / _DESV
    return x.transpose(2, 0, 1)[None]  # forma (1, 3, 224, 224)


def vector(ruta_o_imagen):
    """Vector de 1000 características de una foto."""
    sesion = _obtener_sesion()
    nombre_entrada = sesion.get_inputs()[0].name
    salida = sesion.run(None, {nombre_entrada: _preparar(ruta_o_imagen)})[0]
    return np.asarray(salida, dtype=np.float32).reshape(-1)[:DIMENSION]


COLUMNAS_CNN = [f"cnn_{i}" for i in range(DIMENSION)]
