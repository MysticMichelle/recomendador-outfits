"""
PASO 5 · WEB APP + TABLERO ANALÍTICO
------------------------------------
Pestañas:
  1. Recomendador : el usuario elige una OCASIÓN y sube fotos. Para cada foto,
                    el modelo reconoce el tipo de prenda (con % de seguridad),
                    el color (k-means + reglas o modelo) y sugiere ocasiones.
                    Las reglas de color + la colorimetría de la ocasión
                    califican cada outfit.
  2. Tablero      : KPIs del dataset y de los modelos + gráficas del EDA.
  3. Mi clóset    : prendas guardadas del usuario (se borran solas tras DIAS_GUARDADO días).
  4. Arquitectura : cómo funciona la solución.

Apariencia: .streamlit/config.toml (colores) y assets/estilo.css (estilos extra).

Ejecutar:
    py -3.13 -m streamlit run app.py
"""

import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.caracteristicas import extraer
from src.color_kmeans import rgb_a_hex
from src.closet import obtener_almacen
from src.config import (CLASES, COLORES, DATASET, DIAS_GUARDADO, EDA, ESTILO_CSS, METRICAS, METRICAS_COLOR, METRICAS_OCASION, MODELADO,
                        MODELO_FINAL, NEUTROS, NOMBRE_CLASE, NOMBRE_OCASION, OCASIONES, PATRONES, TONO_COLOR,
                        UMBRAL_SEGURIDAD)
from src.reglas import recomendar

st.set_page_config(page_title="Recomendador de outfits", page_icon="👕", layout="wide")
if ESTILO_CSS.exists():
    st.markdown(f"<style>{ESTILO_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

almacen = obtener_almacen()


@st.cache_resource
def limpiar_al_iniciar():
    """Al abrir la app, borra las prendas guardadas que ya vencieron."""
    return almacen.limpiar_vencidas()


limpiar_al_iniciar()


@st.cache_resource
def cargar_modelos(version):
    """'version' es la fecha de modificación del archivo: si se reentrena o se sube un
    modelo nuevo, cambia y Streamlit vuelve a cargarlo en lugar de usar el de la memoria."""
    return joblib.load(MODELO_FINAL) if MODELO_FINAL.exists() else None


@st.cache_data(show_spinner=False)
def analizar(datos):
    """Extrae características (incluye color con k-means) de una foto subida."""
    return extraer(Image.open(io.BytesIO(datos)))


M = cargar_modelos(MODELO_FINAL.stat().st_mtime if MODELO_FINAL.exists() else 0)


def predecir(f):
    """Tipo (+ seguridad), color y ocasiones sugeridas para una foto."""
    x = pd.DataFrame([f])[M["columnas"]].to_numpy() if M else None
    tipo, seguridad = "superior", None
    if M:
        mt = M["tipo"]["modelo"]
        tipo = str(mt.predict(x)[0])
        if M["tipo"]["con_probabilidad"]:
            seguridad = float(mt.predict_proba(x)[0].max())
    color = f["color_nombre"]
    if M and M.get("color"):
        color = str(M["color"]["modelo"].predict(x)[0])
    ocas = []
    if M and M.get("ocasion"):
        onehot = np.array([[int(tipo == c) for c in CLASES]])
        modelo_oc = M["ocasion"]["modelo"]
        umbrales = M["ocasion"].get("umbrales")
        if umbrales:  # umbral propio por ocasión (calculado en 4_modelado.py)
            probas = [p[0, 1] for p in modelo_oc.predict_proba(np.hstack([x, onehot]))]
            ocas = [o for o, p in zip(M["ocasion"]["ocasiones"], probas) if p >= umbrales[o]]
        else:
            pred = modelo_oc.predict(np.hstack([x, onehot]))[0]
            ocas = [o for o, v in zip(M["ocasion"]["ocasiones"], pred) if v == 1]
    return tipo, seguridad, color, ocas


def color_para_reglas(f, nombre):
    """Arma el color que usan las reglas a partir del nombre elegido."""
    medido = nombre == f["color_nombre"]
    tono = f["dom_tono"] if medido else TONO_COLOR.get(nombre, f["dom_tono"])
    neutro = nombre in NEUTROS
    return {"nombre": nombre, "tono": float(tono), "saturacion": float(f["dom_saturacion"]),
            "brillo": float(f["dom_brillo"]), "neutro": bool(neutro),
            "fuerte": bool((not neutro) and nombre != "multicolor" and f["es_fuerte"])}


# ---------------------------------------------------------------- usuario (Mi clóset)
with st.sidebar:
    st.header("👚 Mi clóset")
    usuario = st.text_input("Tu nombre de usuario", placeholder="ej. michelle",
                            help="Solo letras y números. Demo: no tiene contraseña.")
    st.caption(f"Las prendas que guardes se conservan {DIAS_GUARDADO} días y luego se borran solas.")


def prendas_guardadas():
    if not usuario:
        return []
    try:
        return almacen.listar(usuario)
    except ValueError:
        return []


st.title("👕 Recomendador de outfits")
tab_rec, tab_tablero, tab_closet, tab_arq = st.tabs(["✨ Recomendador", "📊 Tablero analítico", "👚 Mi clóset",
                                                     "🧩 Arquitectura"])

# ===========================================================================
# 1. RECOMENDADOR
# ===========================================================================
with tab_rec:
    if M is None:
        st.warning("Todavía no hay modelos entrenados (corre `py -3.13 4_modelado.py`). "
                   "Mientras, elige todo a mano.")
    else:
        st.caption(f"Tipo de prenda: **{M['tipo']['nombre']}** · Color: "
                   f"**{'Random Forest de color' if M.get('color') else 'k-means + reglas HSV'}** · "
                   f"Ocasiones: **{M['ocasion'].get('nombre', 'Random Forest') + ' (multi-etiqueta)' if M.get('ocasion') else 'a mano'}**")

    opciones = ["todas"] + OCASIONES
    ocasion = st.radio("1 · ¿Para qué ocasión te vas a vestir?", opciones, horizontal=True,
                       format_func=lambda o: "Cualquiera" if o == "todas" else NOMBRE_OCASION[o])
    ocasion = None if ocasion == "todas" else ocasion

    archivos = st.file_uploader("2 · Sube fotos de tu ropa (todas juntas)",
                                type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    prendas = []
    if archivos:
        st.write("3 · Revisa lo que detectó el sistema y corrige si hace falta.")
        cols = st.columns(4)
        for i, archivo in enumerate(archivos):
            datos = archivo.getvalue()
            f = analizar(datos)
            tipo_p, seguridad, color_p, ocas_p = predecir(f)
            k = f"{i}_{archivo.name}"
            with cols[i % 4]:
                st.image(datos)
                rgb = (int(f["dom_r"]), int(f["dom_g"]), int(f["dom_b"]))
                st.markdown(f"<div style='display:flex;gap:8px;align-items:center'><div style='width:24px;"
                            f"height:24px;border-radius:6px;border:1px solid #999;background:{rgb_a_hex(rgb)}'>"
                            f"</div><span>color detectado con k-means</span></div>", unsafe_allow_html=True)
                tipo = st.selectbox("Tipo", CLASES, index=CLASES.index(tipo_p), format_func=NOMBRE_CLASE.get,
                                    key=f"t_{k}")
                if seguridad is not None:
                    if seguridad < UMBRAL_SEGURIDAD:
                        st.warning(f"⚠️ Solo {100 * seguridad:.0f} % seguro del tipo. Revísalo.")
                    else:
                        st.caption(f"✔ {100 * seguridad:.0f} % seguro del tipo")
                color = st.selectbox("Color", COLORES, index=COLORES.index(color_p) if color_p in COLORES else 0,
                                     key=f"c_{k}")
                patron = st.selectbox("Patrón", PATRONES, key=f"p_{k}")
                mezclilla = st.checkbox("Mezclilla", key=f"m_{k}")
                ocas = st.multiselect("Ocasiones", OCASIONES, default=ocas_p, format_func=NOMBRE_OCASION.get,
                                      key=f"o_{k}")
            prendas.append({"nombre": archivo.name, "imagen": datos, "tipo": tipo, "patron": patron,
                            "mezclilla": mezclilla, "ocasiones": ocas, "color": color_para_reglas(f, color)})

    if prendas:
        if st.button("💾 Guardar estas prendas en mi clóset", disabled=not usuario,
                     help=None if usuario else "Escribe tu usuario en la barra lateral"):
            for p in prendas:
                almacen.guardar(usuario, p["nombre"], p["imagen"],
                                {k: p[k] for k in ("tipo", "patron", "mezclilla", "ocasiones", "color")})
            st.success(f"Se guardaron {len(prendas)} prendas por {DIAS_GUARDADO} días.")

    guardadas = prendas_guardadas()
    usar_closet = st.checkbox(f"Incluir mi clóset guardado ({len(guardadas)} prendas)",
                              value=bool(guardadas), disabled=not guardadas)
    top = st.slider("¿Cuántos outfits mostrar?", 1, 15, 5)
    if st.button("✨ Recomendar outfits", type="primary"):
        todas = prendas + ([{"nombre": g["nombre"], "imagen": g["imagen"], **g["datos"]} for g in guardadas]
                           if usar_closet else [])
        resultados = recomendar(todas, ocasion, top=top)
        if not resultados:
            st.warning("No se pudo armar ningún outfit. Se necesita una prenda de arriba y una de abajo, "
                       "o una prenda completa y calzado" +
                       (f", marcadas para «{NOMBRE_OCASION[ocasion]}»." if ocasion else "."))
        for n, r in enumerate(resultados, start=1):
            with st.container(border=True):
                st.markdown(f"<span class='calif'>#{n} · {r['calificacion']} / 10</span>",
                            unsafe_allow_html=True)
                cols = st.columns(len(r["piezas"]) + 1)
                for c, p in zip(cols, r["piezas"]):
                    c.image(p["imagen"], caption=f"{NOMBRE_CLASE[p['tipo']]} · {p['color']['nombre']}")
                with cols[-1]:
                    st.progress(r["calificacion"] / 10)
                    st.markdown("**¿Por qué?**\n" + "\n".join(f"- {x}" for x in r["razones"]))

# ===========================================================================
# 2. TABLERO ANALÍTICO
# ===========================================================================
with tab_tablero:
    if not DATASET.exists():
        st.info("Corre `py -3.13 2_limpieza.py` para generar el dataset.")
    else:
        df = pd.read_csv(DATASET)
        met = pd.read_csv(METRICAS) if METRICAS.exists() else None
        met_c = pd.read_csv(METRICAS_COLOR) if METRICAS_COLOR.exists() else None
        met_o = pd.read_csv(METRICAS_OCASION) if METRICAS_OCASION.exists() and METRICAS_OCASION.stat().st_size > 2 else None

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Fotos en el dataset", len(df))
        k2.metric("Personas", df["persona"].nunique())
        k3.metric("% prendas neutras", f"{100 * df['color'].isin(NEUTROS).mean():.0f}%")
        if met is not None:
            k4.metric(f"Accuracy tipo ({met.iloc[0]['modelo']})", f"{100 * met.iloc[0]['accuracy']:.0f}%")
            k5.metric("F1 macro tipo", f"{met.iloc[0]['f1']:.2f}")
        if met_c is not None:
            k6.metric("Accuracy del color", f"{100 * met_c['accuracy'].max():.0f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Fotos por tipo de prenda**")
            st.bar_chart(df["tipo"].value_counts().reindex(CLASES, fill_value=0))
        with c2:
            st.markdown("**Color real (etiquetado)**")
            st.bar_chart(df["color"].value_counts())
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Prendas por ocasión**")
            st.bar_chart(pd.Series({NOMBRE_OCASION[o]: int(df[f"oc_{o}"].sum()) for o in OCASIONES}))
        with c4:
            st.markdown("**Patrón**")
            st.bar_chart(df["patron"].value_counts())
        st.markdown("**Fotos por persona**")
        st.bar_chart(df["persona"].value_counts())

        if met is not None:
            st.markdown("### Modelos de tipo de prenda")
            st.dataframe(met, hide_index=True)
            st.bar_chart(met.set_index("modelo")[["accuracy", "f1"]], stack=False)
            if (MODELADO / "03_matriz_confusion.png").exists():
                st.image(str(MODELADO / "03_matriz_confusion.png"), caption="Matriz de confusión del mejor modelo")
            if (MODELADO / "01_kmeans_codo_silueta.png").exists():
                st.image(str(MODELADO / "01_kmeans_codo_silueta.png"), caption="No supervisado: elección de k")
        cc, co = st.columns(2)
        if met_c is not None:
            with cc:
                st.markdown("### Color: reglas vs modelo")
                st.dataframe(met_c, hide_index=True)
        if met_o is not None:
            with co:
                st.markdown("### Ocasiones: F1 por ocasión")
                st.dataframe(met_o, hide_index=True)

        graficas = sorted(EDA.glob("*.png")) if EDA.exists() else []
        if graficas:
            st.markdown("### Análisis exploratorio (EDA)")
            cols = st.columns(2)
            for i, g in enumerate(graficas):
                cols[i % 2].image(str(g))

# ===========================================================================
# 3. MI CLÓSET
# ===========================================================================
with tab_closet:
    if not usuario:
        st.info("Escribe tu nombre de usuario en la barra lateral para ver tu clóset.")
    else:
        guardadas = prendas_guardadas()
        st.write(f"**{len(guardadas)}** prendas guardadas para **{usuario}**.")
        cols = st.columns(4)
        for i, g in enumerate(guardadas):
            d = g["datos"]
            with cols[i % 4]:
                st.image(g["imagen"])
                st.markdown(f"{NOMBRE_CLASE[d['tipo']]} · {d['color']['nombre']}  \n"
                            f"<span class='vence'>Se borra el {g['vence'][:10]}</span>", unsafe_allow_html=True)
                if st.button("🗑️ Quitar", key=f"del_{g['id']}"):
                    almacen.borrar(usuario, g["id"])
                    st.rerun()

# ===========================================================================
# 4. ARQUITECTURA
# ===========================================================================
with tab_arq:
    st.markdown("""
### ¿Cómo funciona?

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `1_etiquetar.py` | Etiquetar cada foto: tipo, color real, patrón, mezclilla y ocasiones |
| 2 | `2_limpieza.py` | Homologa etiquetas con regex, quita duplicados y fotos dañadas, extrae características e imputa nulos |
| 3 | `3_eda.py` | Análisis univariado, bivariado y multivariado; qué tanto aciertan las reglas de color |
| 4 | `4_modelado.py` | **No supervisado:** k-means. **Supervisado:** 6 modelos de tipo, modelo de color y modelo de ocasiones |
| 5 | `app.py` | Esta web app: recomendador por ocasión + tablero de KPIs + Mi clóset |

**Flujo en el recomendador:**
ocasión elegida → **filtro**: solo prendas de esa ocasión → para cada foto: k-means (color) → características →
modelo (tipo + % de seguridad) → reglas de teoría del color + **colorimetría de la ocasión** → outfits del 0 al 10.

**Tipos de outfit:** arriba + abajo (+ calzado), o completo (vestido, overol) + calzado.
""")
