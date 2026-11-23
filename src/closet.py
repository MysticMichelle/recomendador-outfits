"""
closet.py
---------
"Mi clóset": guarda las prendas de cada usuario por un tiempo limitado.

La app solo usa estas 4 funciones del almacén:
    guardar(usuario, nombre, imagen_bytes, datos)  -> id
    listar(usuario)                                -> [ {id, nombre, imagen, datos, vence}, ... ]
    borrar(usuario, id)
    limpiar_vencidas()

Hoy se usa AlmacenLocal (carpeta datos/closets/ en la computadora).
Para guardar EN LA NUBE basta con escribir otra clase con esas mismas
4 funciones (por ejemplo con Supabase Storage o Firebase Storage) y
cambiar ALMACEN en src/config.py. La app no necesita otros cambios.
"""

import json
import re
import uuid
from datetime import datetime, timedelta

from src.config import ALMACEN, CLOSETS, DIAS_GUARDADO


def _usuario_seguro(usuario):
    """Solo letras, números, guion y guion bajo (evita rutas raras)."""
    u = re.sub(r"[^a-zA-Z0-9_-]", "", (usuario or "").strip().lower())
    if not u:
        raise ValueError("Escribe un nombre de usuario (solo letras y números).")
    return u


class AlmacenLocal:
    """Guarda cada prenda como datos/closets/<usuario>/<id>.jpg + <id>.json"""

    def __init__(self, carpeta=CLOSETS, dias=DIAS_GUARDADO):
        self.carpeta = carpeta
        self.dias = dias

    def guardar(self, usuario, nombre, imagen_bytes, datos):
        u = _usuario_seguro(usuario)
        carpeta = self.carpeta / u
        carpeta.mkdir(parents=True, exist_ok=True)
        pid = uuid.uuid4().hex[:12]
        ahora = datetime.now()
        (carpeta / f"{pid}.jpg").write_bytes(imagen_bytes)
        meta = {"id": pid, "nombre": nombre, "guardado": ahora.isoformat(timespec="seconds"),
                "vence": (ahora + timedelta(days=self.dias)).isoformat(timespec="seconds"), "datos": datos}
        (carpeta / f"{pid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return pid

    def listar(self, usuario):
        u = _usuario_seguro(usuario)
        carpeta = self.carpeta / u
        if not carpeta.exists():
            return []
        salida = []
        for archivo in sorted(carpeta.glob("*.json")):
            meta = json.loads(archivo.read_text(encoding="utf-8"))
            if datetime.fromisoformat(meta["vence"]) < datetime.now():
                continue
            img = carpeta / f"{meta['id']}.jpg"
            if img.exists():
                salida.append({**meta, "imagen": img.read_bytes()})
        return salida

    def borrar(self, usuario, pid):
        carpeta = self.carpeta / _usuario_seguro(usuario)
        for ext in (".jpg", ".json"):
            (carpeta / f"{re.sub(r'[^a-f0-9]', '', pid)}{ext}").unlink(missing_ok=True)

    def limpiar_vencidas(self):
        """Borra las prendas cuyo tiempo de guardado ya pasó. Regresa cuántas borró."""
        if not self.carpeta.exists():
            return 0
        n = 0
        for archivo in self.carpeta.glob("*/*.json"):
            meta = json.loads(archivo.read_text(encoding="utf-8"))
            if datetime.fromisoformat(meta["vence"]) < datetime.now():
                archivo.with_suffix(".jpg").unlink(missing_ok=True)
                archivo.unlink(missing_ok=True)
                n += 1
        return n


class AlmacenNube:
    """
    PENDIENTE (para quien continúe el proyecto): misma interfaz que AlmacenLocal,
    pero guardando en un servicio en la nube. Ejemplo con Supabase Storage:
      - guardar: subir la imagen a un bucket "closets/<usuario>/<id>.jpg" y
        guardar 'datos' y 'vence' en una tabla "prendas".
      - listar: consultar la tabla filtrando usuario y vence > ahora.
      - borrar: borrar la fila y el archivo.
      - limpiar_vencidas: borrar filas/archivos con vence < ahora
        (o programarlo como tarea diaria en el servicio).
    Las claves del servicio van en .streamlit/secrets.toml, NUNCA en el código.
    """

    def guardar(self, usuario, nombre, imagen_bytes, datos):
        raise NotImplementedError("Falta implementar el almacén en la nube.")

    def listar(self, usuario):
        raise NotImplementedError("Falta implementar el almacén en la nube.")

    def borrar(self, usuario, pid):
        raise NotImplementedError("Falta implementar el almacén en la nube.")

    def limpiar_vencidas(self):
        raise NotImplementedError("Falta implementar el almacén en la nube.")


def obtener_almacen():
    return AlmacenNube() if ALMACEN == "nube" else AlmacenLocal()
