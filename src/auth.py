"""
AUTENTICACIÓN DE USUARIOS CON SUPABASE AUTH
------------------------------------------
Maneja inicio de sesión, registro y cierre de sesión con correo y contraseña,
manteniendo la sesión activa dentro de `st.session_state`.
"""

import hashlib
import os
import re
import streamlit as st


def _obtener_config_supabase() -> tuple[str, str]:
    url = (os.environ.get("SUPABASE_URL") or "").strip()
    key = (os.environ.get("SUPABASE_KEY") or "").strip()
    if not url or not key:
        try:
            cfg = st.secrets.get("supabase", {})
            url = url or (cfg.get("url") or "").strip()
            key = key or (cfg.get("key") or "").strip()
        except Exception:
            pass
    return url, key


def obtener_cliente_auth():
    """Regresa un cliente de Supabase listo para auth, o None si no hay credenciales."""
    url, key = _obtener_config_supabase()
    if not url or not key or url.startswith("PEGA_AQUI"):
        return None
    from supabase import create_client

    return create_client(url, key)


def clave_closet_desde_email(email: str) -> str:
    """Genera un identificador válido para `_validar_usuario` a partir del correo."""
    limpio = re.sub(r"[^a-zA-Z0-9_-]", "_", (email or "").strip().lower())[:24].strip("_") or "usr"
    digest = hashlib.sha256((email or "").strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{limpio}_{digest}"[:40]


def sesion_actual() -> dict | None:
    """Devuelve la sesión guardada en `st.session_state` o None si no hay sesión."""
    sesion = st.session_state.get("auth_sesion")
    if isinstance(sesion, dict) and sesion.get("email"):
        return sesion
    return None


def iniciar_sesion(email: str, password: str) -> tuple[bool, str]:
    """Inicia sesión con correo y contraseña en Supabase Auth."""
    correo = (email or "").strip()
    clave = password or ""
    if not correo or not clave:
        return False, "Escribe tu correo y tu contraseña."

    cliente = obtener_cliente_auth()
    if cliente is None:
        return False, "Falta configurar `[supabase]` en `.streamlit/secrets.toml`."

    try:
        res = cliente.auth.sign_in_with_password({"email": correo, "password": clave})
        user = getattr(res, "user", None)
        session = getattr(res, "session", None)
        if not user or not getattr(user, "email", None):
            return False, "No se pudo iniciar sesión. Revisa tu correo y contraseña."

        st.session_state["auth_sesion"] = {
            "user_id": str(getattr(user, "id", "")),
            "email": str(user.email),
            "closet_key": clave_closet_desde_email(str(user.email)),
            "access_token": getattr(session, "access_token", None) if session else None,
        }
        return True, "Sesión iniciada."
    except Exception as exc:
        mensaje = str(exc)
        if "Invalid login credentials" in mensaje:
            return False, "Correo o contraseña incorrectos."
        if "Email not confirmed" in mensaje:
            return False, "Tu correo aún no está confirmado. Revisa tu bandeja o desactiva «Confirm email» en Supabase Auth."
        return False, f"No se pudo iniciar sesión: {mensaje}"


def registrar_usuario(email: str, password: str) -> tuple[bool, str]:
    """Registra un usuario nuevo con correo y contraseña en Supabase Auth."""
    correo = (email or "").strip()
    clave = password or ""
    if not correo or len(clave) < 6:
        return False, "Ingresa un correo válido y una contraseña de al menos 6 caracteres."

    cliente = obtener_cliente_auth()
    if cliente is None:
        return False, "Falta configurar `[supabase]` en `.streamlit/secrets.toml`."

    try:
        res = cliente.auth.sign_up({"email": correo, "password": clave})
        user = getattr(res, "user", None)
        session = getattr(res, "session", None)

        if user and getattr(user, "email", None) and session:
            st.session_state["auth_sesion"] = {
                "user_id": str(getattr(user, "id", "")),
                "email": str(user.email),
                "closet_key": clave_closet_desde_email(str(user.email)),
                "access_token": getattr(session, "access_token", None),
            }
            return True, "Cuenta creada y sesión iniciada."

        if user and getattr(user, "email", None):
            return True, "Cuenta creada. Revisa tu correo para confirmar tu cuenta y luego inicia sesión."

        return False, "No se pudo completar el registro."
    except Exception as exc:
        return False, f"No se pudo registrar: {exc}"


def cerrar_sesion() -> None:
    """Cierra la sesión actual de Supabase Auth y limpia el estado local."""
    cliente = obtener_cliente_auth()
    if cliente is not None:
        try:
            cliente.auth.sign_out()
        except Exception:
            pass
    st.session_state.pop("auth_sesion", None)