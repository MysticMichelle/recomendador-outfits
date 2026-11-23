"""
PASO 1 · ETIQUETAR LAS FOTOS
----------------------------
App para que cada integrante etiquete sus fotos.
Por cada foto se marca: tipo, color real, patrón, si es mezclilla y ocasiones.
Genera datos/etiquetas/etiquetas_<persona>.csv

Ejecutar:
    py -3.13 -m streamlit run 1_etiquetar.py
"""

import csv
from datetime import datetime

import streamlit as st
from PIL import Image, ImageOps

from src.config import (CLASES, COLORES, DESCRIPCION_CLASE, ETIQUETAS, FOTOS, NOMBRE_CLASE,
                        NOMBRE_OCASION, OCASIONES, PATRONES, listar_fotos)

st.set_page_config(page_title="Etiquetar fotos", page_icon="🏷️", layout="wide")
st.title("🏷️ Etiquetar fotos de ropa")

fotos = listar_fotos()
if not fotos:
    st.error(f"No hay fotos. Cópialas a: {FOTOS}\\<Persona>\\foto.jpg")
    st.stop()

personas = sorted({f.split("/")[0] for f in fotos if "/" in f})
persona = st.selectbox("¿De quién son las fotos que vas a etiquetar?", personas)
fotos_persona = [f for f in fotos if f.startswith(persona + "/")]

ETIQUETAS.mkdir(parents=True, exist_ok=True)
archivo_csv = ETIQUETAS / f"etiquetas_{persona}.csv"
CAMPOS = ["archivo", "persona", "tipo", "color", "patron", "mezclilla", "ocasiones", "fecha"]


def leer():
    if not archivo_csv.exists():
        return []
    with open(archivo_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def escribir(filas):
    with open(archivo_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


filas = leer()
hechas = {r["archivo"] for r in filas}
pendientes = [f for f in fotos_persona if f not in hechas]
n_hechas = len(fotos_persona) - len(pendientes)

st.progress(n_hechas / len(fotos_persona), text=f"{n_hechas} de {len(fotos_persona)} etiquetadas")

if filas and st.button("↩️ Deshacer la última"):
    escribir(filas[:-1])
    st.rerun()

if not pendientes:
    st.success("¡Terminaste todas las fotos de esta persona! 🎉")
    st.stop()

actual = pendientes[0]
img = ImageOps.exif_transpose(Image.open(FOTOS / actual)).convert("RGB")
img.thumbnail((600, 600))


def guardar(tipo, color="", patron="", mezclilla=False, ocasiones=()):
    filas.append({"archivo": actual, "persona": persona, "tipo": tipo, "color": color,
                  "patron": patron, "mezclilla": int(mezclilla), "ocasiones": "|".join(ocasiones),
                  "fecha": datetime.now().isoformat(timespec="seconds")})
    escribir(filas)


col_foto, col_form = st.columns([2, 3])
with col_foto:
    st.image(img, caption=actual)
    if st.button("🗑️ Descartar esta foto", key=f"desc_{actual}"):
        guardar("descartar")
        st.rerun()
    st.caption("Descartar = foto borrosa, sin ropa o con muchas prendas revueltas.")

with col_form:
    with st.form(key=f"form_{actual}"):
        tipo = st.radio("1 · ¿Qué prenda es? (si hay varias, la que ocupa más espacio)", CLASES,
                        index=None, horizontal=True,
                        format_func=lambda c: f"{NOMBRE_CLASE[c]}")
        st.caption(" · ".join(f"{NOMBRE_CLASE[c]}: {DESCRIPCION_CLASE[c]}" for c in CLASES))
        color = st.radio("2 · Color principal real", COLORES, index=None, horizontal=True)
        patron = st.radio("3 · Patrón", PATRONES, index=0, horizontal=True)
        mezclilla = st.checkbox("4 · Es de mezclilla (tela de jean)")
        st.write("5 · ¿Para qué ocasiones sirve? (puedes marcar varias)")
        cols = st.columns(len(OCASIONES))
        marcadas = [o for o, c in zip(OCASIONES, cols) if c.checkbox(NOMBRE_OCASION[o], key=f"oc_{o}_{actual}")]
        enviado = st.form_submit_button("✔ Guardar y siguiente", type="primary")

    if enviado:
        faltan = []
        if tipo is None:
            faltan.append("el tipo de prenda")
        if color is None:
            faltan.append("el color")
        if not marcadas and tipo != "otro":
            faltan.append("al menos una ocasión")
        if faltan:
            st.error("Falta elegir: " + ", ".join(faltan))
        else:
            guardar(tipo, color, patron, mezclilla, marcadas)
            st.rerun()
