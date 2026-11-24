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
    python3 -m streamlit run app.py
"""

import io

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.caracteristicas import extraer
from src.color_kmeans import rgb_a_hex
from src.closet import obtener_almacen
from src.config import (CLASES, COLORES, DATASET, DIAS_GUARDADO, EDA, ESTILO_CSS, METRICAS, METRICAS_COLOR,
                        METRICAS_OCASION, MODELADO, MODELO_FINAL, NEUTROS, NOMBRE_CLASE, NOMBRE_OCASION,
                        OCASIONES, PATRONES, TONO_COLOR, UMBRAL_SEGURIDAD)
from src.reglas import recomendar

MAX_FOTOS_RECOMENDADAS = 30

SVG_PERCHA = """<svg class="svg-icon" width="22" height="22" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<path d="M12 6a2 2 0 1 0-2-2"/><path d="M12 6v2.2"/><path d="M12 8.2 3.5 14.5a1.5 1.5 0 0 0 .9 2.7h15.2a1.5 1.5 0 0 0 .9-2.7L12 8.2z"/>
</svg>"""

SVG_ARMA_CLOSET = """<svg class="svg-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<rect x="4" y="3" width="16" height="18" rx="2"/><line x1="12" y1="3" x2="12" y2="21"/><line x1="9" y1="10" x2="9" y2="13"/><line x1="15" y1="10" x2="15" y2="13"/>
</svg>"""

SVG_ETIQUETA = """<svg class="svg-icon" width="16" height="16" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><circle cx="7" cy="7" r="1.5"/>
</svg>"""

SVG_ALERTA = """<svg class="svg-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
</svg>"""

SVG_OCASION = """<svg class="svg-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><line x1="12" y1="3" x2="12" y2="5"/><line x1="19" y1="12" x2="21" y2="12"/>
</svg>"""

SVG_GUARDAR = """<svg class="svg-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>
</svg>"""

SVG_RAPIDO = """<svg class="svg-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
</svg>"""

SVG_BOMBILLA = """<svg class="svg-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<path d="M9 18h6"/><path d="M10 22h4"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/>
</svg>"""

SVG_ESTRELLA = """<svg class="svg-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
</svg>"""

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


@st.cache_data(show_spinner=False)
def crear_miniatura(datos, lado_max=320):
    """Genera una miniatura ligera en JPEG para mostrar rápido en pantalla sin alentar la app."""
    img = Image.open(io.BytesIO(datos))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((lado_max, lado_max))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


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
        if umbrales:
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
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;font-weight:700;font-size:1.1rem;color:#0f172a;'>"
        f"{SVG_ARMA_CLOSET}<span>Mi clóset</span></div>",
        unsafe_allow_html=True,
    )
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


st.markdown(
    f"""
    <div class="hero-closet">
      <div class="hero-pill">{SVG_PERCHA} <span>Tu armario, con nuevas ideas</span></div>
      <h1 class="hero-titulo">Tu armario ya tiene la respuesta</h1>
      <p class="hero-sub">
        Sube tus prendas, elige la ocasión y descubre combinaciones explicadas en segundos.
        Sin registro obligatorio, o con guardado por {DIAS_GUARDADO} días al escribir tu usuario en Mi clóset.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_rec, tab_tablero, tab_closet, tab_arq = st.tabs([
    "Recomendador",
    "Tablero analítico",
    "Mi clóset",
    "Arquitectura",
])

# ===========================================================================
# 1. RECOMENDADOR
# ===========================================================================
with tab_rec:
    if M is None:
        st.warning("Todavía no hay modelos entrenados (corre `python3 4_modelado.py`). "
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

    if archivos and len(archivos) > MAX_FOTOS_RECOMENDADAS:
        st.markdown(
            f"<div class='aviso-limite'>{SVG_ALERTA}<div>Subiste <strong>{len(archivos)} fotos</strong>. "
            f"Para que la página cargue rápido y sea fácil revisar tus prendas, te recomendamos subir "
            f"máximo <strong>{MAX_FOTOS_RECOMENDADAS} fotos</strong> por tanda (o guardarlas por partes en «Mi clóset»).</div></div>",
            unsafe_allow_html=True,
        )

    prendas = []
    if archivos:
        st.write(f"3 · Revisa lo que detectó el sistema ({len(archivos)} fotos) y corrige si hace falta.")
        cols = st.columns(4)
        for i, archivo in enumerate(archivos):
            datos = archivo.getvalue()
            miniatura = crear_miniatura(datos, lado_max=300)
            f = analizar(datos)
            tipo_p, seguridad, color_p, ocas_p = predecir(f)
            k = f"{i}_{archivo.name}"
            with cols[i % 4]:
                with st.container(border=True):
                    st.image(miniatura, width=190)
                    rgb = (int(f["dom_r"]), int(f["dom_g"]), int(f["dom_b"]))
                    st.markdown(
                        f"<div class='chip-color'><span class='swatch-color' style='background:{rgb_a_hex(rgb)}'>"
                        f"</span><span>Color detectado (k-means)</span></div>",
                        unsafe_allow_html=True,
                    )
                    tipo = st.selectbox("Tipo", CLASES, index=CLASES.index(tipo_p), format_func=NOMBRE_CLASE.get,
                                        key=f"t_{k}")
                    if seguridad is not None:
                        if seguridad < UMBRAL_SEGURIDAD:
                            st.warning(f"Solo {100 * seguridad:.0f} % seguro del tipo. Revísalo.")
                        else:
                            st.caption(f"{100 * seguridad:.0f} % seguro del tipo")
                    color = st.selectbox("Color", COLORES, index=COLORES.index(color_p) if color_p in COLORES else 0,
                                         key=f"c_{k}")
                    with st.expander("Más detalles (patrón, mezclilla, ocasiones)"):
                        patron = st.selectbox("Patrón", PATRONES, key=f"p_{k}")
                        mezclilla = st.checkbox("Mezclilla", key=f"m_{k}")
                        ocas = st.multiselect("Ocasiones", OCASIONES, default=ocas_p, format_func=NOMBRE_OCASION.get,
                                              key=f"o_{k}")
            prendas.append({"nombre": archivo.name, "imagen": datos, "tipo": tipo, "patron": patron,
                            "mezclilla": mezclilla, "ocasiones": ocas, "color": color_para_reglas(f, color)})

    if prendas:
        if not usuario:
            st.caption("Escribe tu nombre de usuario en el panel izquierdo y presiona Enter para activar el guardado.")
        if st.button("Guardar estas prendas en mi clóset", disabled=not usuario,
                     help=None if usuario else "Escribe tu usuario en la barra lateral"):
            for p in prendas:
                almacen.guardar(usuario, p["nombre"], p["imagen"],
                                {k: p[k] for k in ("tipo", "patron", "mezclilla", "ocasiones", "color")})
            st.success(f"Se guardaron {len(prendas)} prendas por {DIAS_GUARDADO} días.")

    guardadas = prendas_guardadas()
    usar_closet = st.checkbox(f"Incluir mi clóset guardado ({len(guardadas)} prendas)",
                              value=bool(guardadas), disabled=not guardadas)
    top = st.slider("¿Cuántos outfits mostrar?", 1, 15, 5)
    if st.button("Recomendar outfits", type="primary"):
        todas = prendas + ([{"nombre": g["nombre"], "imagen": g["imagen"], **g["datos"]} for g in guardadas]
                           if usar_closet else [])
        resultados_raw = recomendar(todas, ocasion, top=max(top * 3, top))
        resultados = []
        vistas = set()
        for r in resultados_raw:
            firma = tuple(sorted(
                (p.get("nombre", ""), p.get("tipo", ""), p.get("color", {}).get("nombre", ""), len(p.get("imagen", b"")))
                for p in r.get("piezas", [])
            ))
            if firma not in vistas:
                vistas.add(firma)
                resultados.append(r)
            if len(resultados) >= top:
                break
        if not resultados:
            st.warning("No se pudo armar ningún outfit. Se necesita una prenda de arriba y una de abajo, "
                       "o una prenda completa y calzado" +
                       (f", marcadas para «{NOMBRE_OCASION[ocasion]}»." if ocasion else "."))
        else:
            etiqueta_oc = NOMBRE_OCASION[ocasion] if ocasion else "Cualquier ocasión"
            st.markdown(
                f"<div class='resumen-outfits'>{SVG_ETIQUETA}<div>Se muestran <strong>{len(resultados)}</strong> "
                f"{'recomendación' if len(resultados) == 1 else 'recomendaciones'} para "
                f"<strong>{etiqueta_oc}</strong>.</div></div>",
                unsafe_allow_html=True,
            )
        for n, r in enumerate(resultados, start=1):
            with st.container(border=True):
                c_head1, c_head2 = st.columns([2, 3])
                with c_head1:
                    st.markdown(
                        f"<span class='calif'>{SVG_ETIQUETA}<span>Outfit #{n} · {r['calificacion']} / 10</span></span>",
                        unsafe_allow_html=True,
                    )
                with c_head2:
                    st.progress(min(max(r["calificacion"] / 10, 0.0), 1.0))

                cols = st.columns(len(r["piezas"]) + 1)
                for c, p in zip(cols, r["piezas"]):
                    with c:
                        st.image(crear_miniatura(p["imagen"], lado_max=280), width=175)
                        st.markdown(
                            f"<div class='pieza-titulo'>{NOMBRE_CLASE[p['tipo']]}</div>"
                            f"<div class='pieza-sub'>Color: {p['color']['nombre']}</div>",
                            unsafe_allow_html=True,
                        )
                with cols[-1]:
                    st.markdown("**¿Por qué combina?**\n" + "\n".join(f"- {x}" for x in r["razones"]))

    st.divider()
    st.markdown("<div class='acento-linea'></div>", unsafe_allow_html=True)
    st.markdown("### Cómo funciona en 4 pasos")
    p1, p2, p3, p4 = st.columns(4)
    pasos_info = [
        (p1, "01", "Sube tus fotos", "Añade imágenes de las prendas que quieres combinar hoy."),
        (p2, "02", "Elige la ocasión", "Selecciona casual, trabajo, noche o formal según tu plan."),
        (p3, "03", "Decide cuántas", "Ajusta el número máximo de opciones que quieres revisar."),
        (p4, "04", "Revisa y guarda", f"Mira calificación y motivos; con usuario guarda {DIAS_GUARDADO} días."),
    ]
    for col, num, tit, desc in pasos_info:
        with col:
            st.markdown(
                f"<div class='paso-card'><div class='paso-num'>{num}</div>"
                f"<div class='paso-titulo'>{tit}</div><p class='paso-texto'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='acento-linea'></div>", unsafe_allow_html=True)
    st.markdown("### Todo lo que necesitas para vestirte hoy")
    st.caption("Pensado para el día a día: rápido, claro y con tu ropa de siempre.")

    beneficios = [
        (SVG_PERCHA, "Solo tu ropa", "Combina prendas que ya tienes, sin comprar nada nuevo."),
        (SVG_OCASION, "Elige la ocasión", "Recomendaciones ajustadas al momento y al estilo."),
        (SVG_GUARDAR, f"Guarda {DIAS_GUARDADO} días", "Con tu usuario en Mi clóset, tus prendas se conservan por tiempo limitado."),
        (SVG_RAPIDO, "Sin registro previo", "Puedes probar el recomendador al momento y subir fotos directo."),
        (SVG_BOMBILLA, "¿Por qué combina?", "Cada outfit explica las razones de color y ocasión."),
        (SVG_ESTRELLA, "Con calificación", "Puntaje del 0 al 10 para comparar opciones rápido."),
    ]
    fila_a = st.columns(3)
    for idx, (svg_ico, tit, desc) in enumerate(beneficios[:3]):
        with fila_a[idx]:
            st.markdown(
                f"<div class='feature-card'><div class='feature-icon-box'>{svg_ico}</div>"
                f"<div class='feature-titulo'>{tit}</div><p class='feature-texto'>{desc}</p></div>",
                unsafe_allow_html=True,
            )
    fila_b = st.columns(3)
    for idx, (svg_ico, tit, desc) in enumerate(beneficios[3:]):
        with fila_b[idx]:
            st.markdown(
                f"<div class='feature-card'><div class='feature-icon-box'>{svg_ico}</div>"
                f"<div class='feature-titulo'>{tit}</div><p class='feature-texto'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='acento-linea'></div>", unsafe_allow_html=True)
    st.markdown("### Preguntas frecuentes")
    faqs = [
        ("¿Tengo que registrarme para usar el recomendador?",
         "No. Puedes subir tus fotos y generar combinaciones al momento sin escribir usuario."),
        ("¿Qué pasa con mis prendas si escribo un usuario?",
         f"Se guardan en Mi clóset durante {DIAS_GUARDADO} días para reutilizarlas y luego se limpian automáticamente."),
        ("¿Por qué a veces me muestra menos outfits de los que pedí?",
         "El selector indica el máximo a mostrar; si con las prendas actuales y la ocasión elegida solo se forman menos combinaciones válidas, se muestran las disponibles."),
        ("¿Necesito comprar ropa nueva?",
         "No, el recomendador trabaja con las prendas que tú subas o tengas guardadas en tu clóset."),
        ("¿Sirve para cualquier ocasión?",
         "Sí, puedes elegir Cualquiera o filtrar por una ocasión específica."),
    ]
    for pregunta, respuesta in faqs:
        with st.expander(pregunta):
            st.write(respuesta)

# ===========================================================================
# 2. TABLERO ANALÍTICO
# ===========================================================================
with tab_tablero:
    if not DATASET.exists():
        st.info("Corre `python3 2_limpieza.py` para generar el dataset.")
    else:
        df = pd.read_csv(DATASET)
        met = pd.read_csv(METRICAS) if METRICAS.exists() else None
        met_c = pd.read_csv(METRICAS_COLOR) if METRICAS_COLOR.exists() else None
        met_o = pd.read_csv(METRICAS_OCASION) if METRICAS_OCASION.exists() and METRICAS_OCASION.stat().st_size > 2 else None

        def barras(serie, titulo, color="#b91c46", tam_letra=13, alto_fila=32):
            """Barras horizontales ordenadas de mayor a menor, con etiquetas claras y legibles."""
            datos = serie.rename_axis("categoria").reset_index(name="valor")
            grafica = alt.Chart(datos).mark_bar(color=color, cornerRadiusEnd=4).encode(
                x=alt.X("valor:Q", title=None, axis=alt.Axis(labelFontSize=12)),
                y=alt.Y(
                    "categoria:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelLimit=260, labelFontSize=tam_letra, labelPadding=8),
                ),
                tooltip=[alt.Tooltip("categoria:N", title=titulo), alt.Tooltip("valor:Q", title="Cantidad")],
            ).properties(height=max(180, alto_fila * len(datos)), width="container")
            st.markdown(f"#### {titulo}")
            st.altair_chart(grafica)

        integrantes = df.loc[df["persona"].str.lower() != "general", "persona"].nunique()
        st.markdown("#### Datos")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Fotos en el dataset", f"{len(df):,}")
        k2.metric("Integrantes que aportaron fotos", integrantes)
        k3.metric("% prendas neutras", f"{100 * df['color'].isin(NEUTROS).mean():.0f}%")
        k4.metric("% prendas casuales", f"{100 * df['oc_casual'].mean():.0f}%")
        st.markdown("#### Modelos")
        k5, k6, k7, k8 = st.columns(4)
        if met is not None:
            k5.metric(f"Accuracy tipo ({met.iloc[0]['modelo']})", f"{100 * met.iloc[0]['accuracy']:.0f}%")
            k6.metric("F1 macro tipo", f"{met.iloc[0]['f1']:.2f}")
        if met_c is not None:
            k7.metric("Accuracy del color", f"{100 * met_c['accuracy'].max():.0f}%")
        if met_o is not None and "precision" in met_o:
            k8.metric("Precisión de ocasiones", f"{100 * met_o['precision'].mean():.0f}%")

        c1, c2 = st.columns(2)
        with c1:
            barras(df["tipo"].map(lambda t: NOMBRE_CLASE[t].split(" ", 1)[-1]).value_counts(), "Fotos por tipo de prenda")
        with c2:
            barras(df["color"].value_counts(), "Color real (etiquetado)", "#b8892b")
        c3, c4 = st.columns(2)
        with c3:
            barras(pd.Series({NOMBRE_OCASION[o]: int(df[f"oc_{o}"].sum()) for o in OCASIONES}), "Prendas por ocasión")
        with c4:
            barras(df["patron"].value_counts(), "Patrón", "#b8892b")

        barras(df["persona"].value_counts(), "Fotos por carpeta", color="#8f1335", tam_letra=15, alto_fila=38)

        if met is not None:
            st.markdown("### Modelos de tipo de prenda")
            st.dataframe(met, hide_index=True)
            comp = met.melt(id_vars="modelo", value_vars=["accuracy", "f1"], var_name="métrica", value_name="valor")
            st.altair_chart(alt.Chart(comp).mark_bar().encode(
                y=alt.Y("modelo:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelFontSize=13)),
                x=alt.X("valor:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(labelFontSize=12)),
                yOffset="métrica:N",
                color=alt.Color("métrica:N", scale=alt.Scale(range=["#b91c46", "#b8892b"]), title=None),
                tooltip=["modelo", "métrica", alt.Tooltip("valor:Q", format=".3f")],
            ).properties(height=340, width="container"))
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
                st.markdown("### Ocasiones: umbral, precisión, recall y F1")
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
        st.info("Escribe tu nombre de usuario en la barra lateral para ver o guardar tu clóset.")
    else:
        guardadas = prendas_guardadas()
        st.write(f"**{len(guardadas)}** {'prenda guardada' if len(guardadas) == 1 else 'prendas guardadas'} para **{usuario}**.")
        if not guardadas:
            st.info("Aún no tienes prendas guardadas. Ve a **Recomendador**, sube fotos y pulsa **Guardar estas prendas en mi clóset**.")
        else:
            if len(guardadas) > MAX_FOTOS_RECOMENDADAS:
                st.markdown(
                    f"<div class='aviso-limite'>{SVG_ALERTA}<div>Tienes <strong>{len(guardadas)} prendas</strong> guardadas. "
                    f"Las fotos se muestran en miniatura para mantener la página rápida.</div></div>",
                    unsafe_allow_html=True,
                )
            cols = st.columns(4)
            for i, g in enumerate(guardadas):
                d = g["datos"]
                with cols[i % 4]:
                    with st.container(border=True):
                        st.image(crear_miniatura(g["imagen"], lado_max=280), width=185)
                        st.markdown(
                            f"**{NOMBRE_CLASE[d['tipo']]}** · {d['color']['nombre']}  \n"
                            f"<span class='vence'>{SVG_ETIQUETA}<span>Se borra el {g['vence'][:10]}</span></span>",
                            unsafe_allow_html=True,
                        )
                        if st.button("Quitar prenda", key=f"del_{g['id']}"):
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
