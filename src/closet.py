import base64
import json
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import config as _cfg
from src.config import DIAS_GUARDADO

CLOSETS_DIR = getattr(_cfg, "CLOSETS_DIR", Path("datos/closets"))

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_USER_RE = re.compile(r"^[a-zA-Z0-9_-]{1,40}$")


def _validar_usuario(usuario: str) -> str:
    u = (usuario or "").strip()
    if not u or not _USER_RE.match(u):
        raise ValueError("Usuario inválido: usa solo letras, números, guion o guion bajo (máx. 40).")
    return u


def _validar_id(prenda_id: str) -> str:
    pid = (prenda_id or "").strip()
    if not _ID_RE.match(pid):
        raise ValueError("ID de prenda inválido.")
    return pid


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(texto: str) -> datetime:
    dt = datetime.fromisoformat(texto)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class AlmacenLocal:
    """Guarda prendas de usuario en disco y las elimina tras `dias_guardado` días."""

    def __init__(self, base_dir: Path = CLOSETS_DIR, dias_guardado: int = DIAS_GUARDADO):
        self.base_dir = Path(base_dir)
        self.dias_guardado = int(dias_guardado)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _dir_usuario(self, usuario: str, crear: bool = False) -> Path:
        u = _validar_usuario(usuario)
        raiz = self.base_dir.resolve()
        carpeta = (raiz / u).resolve()
        if carpeta != raiz and raiz not in carpeta.parents:
            raise ValueError("Ruta de usuario fuera del directorio permitido.")
        if crear:
            carpeta.mkdir(parents=True, exist_ok=True)
        return carpeta

    def _es_vencida(self, meta: dict, ahora: datetime) -> bool:
        vence = meta.get("vence_en")
        if not vence:
            return False
        try:
            return _parse_iso(vence) <= ahora
        except ValueError:
            return False

    def limpiar_vencidas(self) -> int:
        """Borra todas las prendas cuyo `vence_en` ya pasó. Regresa cuántas borró."""
        if not self.base_dir.exists():
            return 0
        ahora = _ahora()
        borradas = 0
        for item in self.base_dir.iterdir():
            if not item.is_dir():
                continue
            restantes = 0
            for meta_path in item.glob("*.json"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                if self._es_vencida(meta, ahora):
                    img_name = meta.get("imagen")
                    if img_name:
                        img_path = (item / img_name).resolve()
                        if item.resolve() in img_path.parents and img_path.exists():
                            img_path.unlink()
                    meta_path.unlink(missing_ok=True)
                    borradas += 1
                else:
                    restantes += 1
            if restantes == 0 and not any(item.iterdir()):
                item.rmdir()
        return borradas

    def guardar(self, usuario: str, nombre: str, imagen_bytes: bytes, datos: dict) -> str:
        """Guarda imagen + metadatos para `usuario`. Regresa el `id` de la prenda."""
        carpeta = self._dir_usuario(usuario, crear=True)
        prenda_id = uuid.uuid4().hex[:12]
        ext = Path(nombre or "prenda.jpg").suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        img_name = f"{prenda_id}{ext}"
        (carpeta / img_name).write_bytes(imagen_bytes)

        ahora = _ahora()
        vence = ahora + timedelta(days=self.dias_guardado)
        meta = {
            "id": prenda_id,
            "nombre": nombre or img_name,
            "imagen": img_name,
            "creado_en": _iso(ahora),
            "vence_en": _iso(vence),
            "datos": datos,
        }
        (carpeta / f"{prenda_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return prenda_id

    def listar(self, usuario: str) -> list[dict]:
        """Regresa las prendas vigentes de `usuario` (las vencidas se eliminan al vuelo)."""
        carpeta = self._dir_usuario(usuario, crear=False)
        if not carpeta.exists():
            return []
        ahora = _ahora()
        prendas = []
        for meta_path in sorted(carpeta.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if self._es_vencida(meta, ahora):
                img_name = meta.get("imagen")
                if img_name:
                    img_path = carpeta / img_name
                    if img_path.exists():
                        img_path.unlink()
                meta_path.unlink(missing_ok=True)
                continue

            img_path = carpeta / meta.get("imagen", "")
            if not img_path.exists():
                continue
            prendas.append({
                "id": meta["id"],
                "nombre": meta.get("nombre", img_path.name),
                "imagen": img_path.read_bytes(),
                "creado_en": meta.get("creado_en"),
                "vence_en": meta.get("vence_en"),
                "datos": meta.get("datos", {}),
            })
        prendas.sort(key=lambda p: p.get("creado_en") or "", reverse=True)
        return prendas

    def borrar(self, usuario: str, prenda_id: str) -> bool:
        """Elimina una prenda concreta del usuario."""
        carpeta = self._dir_usuario(usuario, crear=False)
        pid = _validar_id(prenda_id)
        if not carpeta.exists():
            return False
        meta_path = carpeta / f"{pid}.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            img_name = meta.get("imagen")
            if img_name:
                img_path = carpeta / img_name
                if img_path.exists():
                    img_path.unlink()
        except Exception:
            pass
        meta_path.unlink(missing_ok=True)
        return True

    def borrar_usuario(self, usuario: str) -> None:
        carpeta = self._dir_usuario(usuario, crear=False)
        if carpeta.exists():
            shutil.rmtree(carpeta)


class AlmacenSupabase:
    """Guarda prendas en la tabla `closet_prendas` de Supabase con vencimiento en días."""

    def __init__(self, url: str, key: str, tabla: str = "closet_prendas", dias_guardado: int = DIAS_GUARDADO):
        from supabase import create_client

        self.client = create_client(url, key)
        self.tabla = tabla
        self.dias_guardado = int(dias_guardado)

    def limpiar_vencidas(self) -> int:
        ahora_iso = _iso(_ahora())
        resp = (
            self.client.table(self.tabla)
            .delete()
            .lte("vence_en", ahora_iso)
            .execute()
        )
        return len(resp.data or [])

    def guardar(self, usuario: str, nombre: str, imagen_bytes: bytes, datos: dict) -> str:
        u = _validar_usuario(usuario)
        prenda_id = uuid.uuid4().hex[:12]
        ahora = _ahora()
        vence = ahora + timedelta(days=self.dias_guardado)
        fila = {
            "id": prenda_id,
            "usuario": u,
            "nombre": nombre or f"{prenda_id}.jpg",
            "imagen_b64": base64.b64encode(imagen_bytes).decode("ascii"),
            "datos": datos,
            "creado_en": _iso(ahora),
            "vence_en": _iso(vence),
        }
        self.client.table(self.tabla).insert(fila).execute()
        return prenda_id

    def listar(self, usuario: str) -> list[dict]:
        u = _validar_usuario(usuario)
        ahora = _ahora()
        resp = (
            self.client.table(self.tabla)
            .select("id, nombre, imagen_b64, datos, creado_en, vence_en")
            .eq("usuario", u)
            .order("creado_en", desc=True)
            .execute()
        )
        prendas = []
        for fila in resp.data or []:
            vence_en = fila.get("vence_en")
            if vence_en:
                try:
                    if _parse_iso(vence_en) <= ahora:
                        self.borrar(u, fila["id"])
                        continue
                except ValueError:
                    pass
            b64 = fila.get("imagen_b64")
            if not b64:
                continue
            try:
                img_bytes = base64.b64decode(b64.encode("ascii"))
            except Exception:
                continue
            datos = fila.get("datos") or {}
            if isinstance(datos, str):
                try:
                    datos = json.loads(datos)
                except Exception:
                    datos = {}
            prendas.append({
                "id": fila["id"],
                "nombre": fila.get("nombre", f"{fila['id']}.jpg"),
                "imagen": img_bytes,
                "creado_en": fila.get("creado_en"),
                "vence_en": vence_en,
                "datos": datos,
            })
        return prendas

    def borrar(self, usuario: str, prenda_id: str) -> bool:
        u = _validar_usuario(usuario)
        pid = _validar_id(prenda_id)
        resp = (
            self.client.table(self.tabla)
            .delete()
            .eq("usuario", u)
            .eq("id", pid)
            .execute()
        )
        return bool(resp.data)


def obtener_almacen():
    """Usa Supabase desde variables de entorno o st.secrets['supabase']; si no, disco local."""
    import os

    url = (os.environ.get("SUPABASE_URL") or "").strip()
    key = (os.environ.get("SUPABASE_KEY") or "").strip()
    tabla = (os.environ.get("SUPABASE_TABLA") or "closet_prendas").strip()

    if not url or not key:
        try:
            import streamlit as st

            cfg = st.secrets.get("supabase", {})
            url = url or (cfg.get("url") or "").strip()
            key = key or (cfg.get("key") or "").strip()
            tabla = (cfg.get("tabla") or tabla or "closet_prendas").strip()
        except Exception:
            pass

    if url and key and not url.startswith("PEGA_AQUI"):
        return AlmacenSupabase(url=url, key=key, tabla=tabla, dias_guardado=DIAS_GUARDADO)
    return AlmacenLocal()