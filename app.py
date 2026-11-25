"""
WEB APP DEL RECOMENDADOR DE OUTFITS
-----------------------------------
Pestañas:
1. Recomendador : el usuario elige una OCASIÓN y sube fotos. Para cada foto,
                    el sistema reconoce el tipo de prenda, el color y sugiere
                    ocasiones, y califica combinaciones.
2. Mi clóset    : prendas guardadas del usuario (se borran solas tras DIAS_GUARDADO días).

Apariencia: .streamlit/config.toml (colores) y assets/estilo.css (estilos extra).

Ejecutar:
    python3 -m streamlit run app.py
"""

import io
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src import cnn
from src.caracteristicas import extraer
from src.color_kmeans import rgb_a_hex
from src.closet import obtener_almacen
from src.config import (CLASES, COLORES, DIAS_GUARDADO, ESTILO_CSS, MODELO_FINAL, NEUTROS, NOMBRE_CLASE,
                        NOMBRE_OCASION, OCASIONES, PATRONES, TONO_COLOR)
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


def limpiar_razon(texto):
    """Quita valores numéricos entre paréntesis al final de cada razón (ej. (+3), (0.46))."""
    limpio = re.sub(r"\s*\([^)]*\)\s*$", "", str(texto)).strip()
    return limpio or str(texto).strip()


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

MODELO_CNN = MODELO_FINAL.parent / "modelos_cnn.joblib"


@st.cache_resource
def cargar_modelos_cnn(version):
    """Modelo con características de la CNN preentrenada (5_cnn_experimento.py).
    Solo se usa si existe el archivo y están instalados onnxruntime y MobileNetV2."""
    return joblib.load(MODELO_CNN) if MODELO_CNN.exists() and cnn.disponible() else None


MC = cargar_modelos_cnn(MODELO_CNN.stat().st_mtime if MODELO_CNN.exists() else 0)


@st.cache_data(show_spinner=False)
def analizar_cnn(datos):
    """Vector de 1000 características de MobileNetV2 para una foto subida."""
    return cnn.vector(Image.open(io.BytesIO(datos)))


def predecir(f, datos=None):
    """Tipo, color y ocasiones sugeridas para una foto.
    Si está el modelo con CNN (MC), el tipo y las ocasiones se predicen con las características
    de MobileNetV2; el color sigue usando k-means + Random Forest de color."""
    x = pd.DataFrame([f])[M["columnas"]].to_numpy() if M else None
    tipo = "superior"
    x_cnn = None
    if MC and M and datos is not None:
        v = analizar_cnn(datos)[None]
        x_cnn = v if MC["entrada"].startswith("CNN (") else np.hstack([v, x])
        tipo = str(MC["tipo"]["modelo"].predict(x_cnn)[0])
    elif M:
        tipo = str(M["tipo"]["modelo"].predict(x)[0])
    color = f["color_nombre"]
    if M and M.get("color"):
        color = str(M["color"]["modelo"].predict(x)[0])
    ocas = []
    fuente = MC if (x_cnn is not None and MC.get("ocasion")) else M
    if fuente and fuente.get("ocasion"):
        onehot = np.array([[int(tipo == c) for c in CLASES]])
        x_oc = np.hstack([x_cnn if fuente is MC else x, onehot])
        modelo_oc = fuente["ocasion"]["modelo"]
        umbrales = fuente["ocasion"].get("umbrales")
        if umbrales:
            probas = [p[0, 1] for p in modelo_oc.predict_proba(x_oc)]
            ocas = [o for o, p in zip(fuente["ocasion"]["ocasiones"], probas) if p >= umbrales[o]]
        else:
            pred = modelo_oc.predict(x_oc)[0]
            ocas = [o for o, val in zip(fuente["ocasion"]["ocasiones"], pred) if val == 1]
    return tipo, color, ocas


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
    <div id="inicio" class="hero-closet">
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

tab_rec, tab_closet = st.tabs([
    "Recomendador",
    "Mi clóset",
])

# ===========================================================================
# 1. RECOMENDADOR
# ===========================================================================
with tab_rec:
    if M is None:
        st.warning("Todavía no hay modelos entrenados. Elige todo a mano.")

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
            tipo_p, color_p, ocas_p = predecir(f, datos)
            k = f"{i}_{archivo.name}"
            with cols[i % 4]:
                with st.container(border=True):
                    st.image(miniatura, width=190)
                    rgb = (int(f["dom_r"]), int(f["dom_g"]), int(f["dom_b"]))
                    st.markdown(
                        f"<div class='chip-color'><span class='swatch-color' style='background:{rgb_a_hex(rgb)}'>"
                        f"</span><span>Color detectado</span></div>",
                        unsafe_allow_html=True,
                    )
                    tipo = st.selectbox("Tipo", CLASES, index=CLASES.index(tipo_p), format_func=NOMBRE_CLASE.get,
                                        key=f"t_{k}")
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
                    razones_limpias = [limpiar_razon(x) for x in r["razones"]]
                    st.markdown("**¿Por qué combina?**\n" + "\n".join(f"- {x}" for x in razones_limpias))

    st.divider()
    st.markdown("<div id='como-funciona' class='acento-linea'></div>", unsafe_allow_html=True)
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
    st.markdown("<div id='beneficios' class='acento-linea'></div>", unsafe_allow_html=True)
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
    b_izq, b_der = st.columns(2)
    with b_izq:
        st.markdown(
            f"""
            <div class='bloque-editorial'>
              <div class='acento-linea'></div>
              <div class='bloque-titulo'>Tu ropa, guardada</div>
              <p class='bloque-texto'>
                Escribe tu usuario en la barra lateral y tus prendas quedan guardadas hasta por {DIAS_GUARDADO} días
                para volver a combinarlas en tu próxima visita.
              </p>
              <ul class='check-lista'>
                <li><span class='check-circulo'>✓</span><span>Guardado hasta por {DIAS_GUARDADO} días</span></li>
                <li><span class='check-circulo'>✓</span><span>Sin volver a subir cada vez</span></li>
                <li><span class='check-circulo'>✓</span><span>O úsalo al instante sin guardar</span></li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b_der:
        st.markdown(
            """
            <div class='bloque-editorial'>
              <div class='acento-linea'></div>
              <div class='bloque-titulo'>Recibe tus outfits</div>
              <p class='bloque-texto'>
                Cada combinación llega con calificación del 0 al 10 y una explicación clara de por qué funciona
                según colorimetría y ocasión.
              </p>
              <ul class='check-lista'>
                <li><span class='check-circulo'>✓</span><span>Calificación visible en cada outfit</span></li>
                <li><span class='check-circulo'>✓</span><span>Sección clara de «¿Por qué combina?»</span></li>
                <li><span class='check-circulo'>✓</span><span>Ideas listas para tu día o evento</span></li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div id='faq' class='acento-linea'></div>", unsafe_allow_html=True)
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

    st.markdown(
        """
        <footer class="footer-oscuro">
          <div>
            <div class="footer-marca">Recomendador de outfits</div>
            <p class="footer-desc">Combinaciones inteligentes con la ropa que ya tienes.</p>
          </div>
          <ul class="footer-links">
            <li><a href="#inicio">Inicio</a></li>
            <li><a href="#como-funciona">Cómo funciona</a></li>
            <li><a href="#beneficios">Beneficios</a></li>
            <li><a href="#faq">Preguntas frecuentes</a></li>
          </ul>
        </footer>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# 2. MI CLÓSET
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