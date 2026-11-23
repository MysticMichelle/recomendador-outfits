# Guía para continuar el desarrollo de la interfaz

Esta guía es para quien vaya a **mejorar el diseño** de la app y **agregar funciones**, sin romper la parte de inteligencia artificial.

## 1. Cómo correr el proyecto

Requiere Python 3.13 (en Windows: `py install 3.13`).

```bash
py -3.13 -m pip install -r requirements.txt
py -3.13 demo/generar_fotos_demo.py     # fotos de prueba (opcional)
py -3.13 2_limpieza.py
py -3.13 4_modelado.py                  # entrena y guarda modelos/modelos.joblib
py -3.13 -m streamlit run app.py        # abre http://localhost:8501
```

Streamlit recarga la página sola cada vez que guardas un archivo, así que puedes ver cada cambio de diseño al instante.

## 2. Qué archivo hace qué

| Archivo | Qué contiene | ¿Se puede modificar? |
|---|---|---|
| `app.py` | **La interfaz**: pestañas, botones, cómo se muestran las fotos y los outfits | Sí, es el archivo principal para el diseño |
| `.streamlit/config.toml` | **Tema**: colores principales, fondo y tipo de letra | Sí |
| `assets/estilo.css` | **CSS extra** que se carga al iniciar la app | Sí |
| `src/closet.py` | **Mi clóset**: guardar, listar y borrar prendas; borrado automático | Sí (aquí va el guardado en la nube) |
| `src/config.py` | Listas de colores, patrones, ocasiones, días de guardado | Con cuidado |
| `src/reglas.py` | Reglas de teoría del color y colorimetría por ocasión | Con cuidado |
| `src/caracteristicas.py`, `src/color_kmeans.py` | Cómo la IA "lee" una foto | **No** (cambiarlo obliga a reentrenar) |
| `1_etiquetar.py` … `4_modelado.py` | Preparación de datos y entrenamiento | **No** para el diseño |

## 3. Las funciones que usa la interfaz

En `app.py`, la parte visual solo llama estas funciones. Si se respeta lo que entra y lo que sale, se puede rediseñar todo lo demás.

| Función | Entra | Sale |
|---|---|---|
| `analizar(bytes_de_la_foto)` | la foto subida | diccionario con 67 características + `color_nombre` |
| `predecir(caracteristicas)` | lo anterior | `(tipo, seguridad 0–1, color, [ocasiones])` |
| `color_para_reglas(caracteristicas, nombre_color)` | lo anterior | diccionario de color para las reglas |
| `recomendar(prendas, ocasion, top)` (en `src/reglas.py`) | lista de prendas (ver abajo) | lista de outfits `{piezas, calificacion, razones}` |
| `almacen.guardar / listar / borrar` (en `src/closet.py`) | usuario, foto y datos | ver sección 5 |

Cada **prenda** que se le pasa a `recomendar` es un diccionario:

```python
{"nombre": "blusa.jpg", "imagen": b"...bytes...", "tipo": "superior",
 "patron": "liso", "mezclilla": False, "ocasiones": ["casual", "business_casual"],
 "color": {"nombre": "rosa", "tono": 330.0, "saturacion": 0.3, "brillo": 0.8, "neutro": False, "fuerte": False}}
```

## 4. Cambiar el diseño

Hay tres niveles, del más fácil al más completo:

1. **Colores y letra** → `.streamlit/config.toml` (`primaryColor`, `backgroundColor`, `font`…).
2. **CSS** → `assets/estilo.css`. La app ya usa las clases `.calif` (calificación) y `.vence` (fecha de borrado). Puedes agregar más con `st.markdown("<div class='mi-clase'>…</div>", unsafe_allow_html=True)`.
3. **Estructura** → `app.py`: `st.columns`, `st.tabs`, `st.container(border=True)`, `st.expander`, etc. Documentación: https://docs.streamlit.io

> Evitar parámetros marcados como obsoletos en Streamlit (por ejemplo `use_container_width`), porque generan *warnings* y la profesora pidió código sin errores ni *warnings*.

## 5. Guardar el clóset en la nube

Hoy "Mi clóset" funciona **en la computadora** (`AlmacenLocal`, carpeta `datos/closets/<usuario>/`). Cada prenda se guarda con su fecha de vencimiento (`DIAS_GUARDADO = 30` en `src/config.py`), y al abrir la app se borran las vencidas.

Para pasarlo a la nube:

1. Elegir un servicio con almacenamiento de archivos, por ejemplo **Supabase Storage** o **Firebase Storage**.
2. Completar la clase `AlmacenNube` en `src/closet.py` con las mismas 4 funciones (`guardar`, `listar`, `borrar`, `limpiar_vencidas`). En el código vienen comentarios con los pasos.
3. Poner las claves del servicio en `.streamlit/secrets.toml` (nunca en el código ni en GitHub) y leerlas con `st.secrets`.
4. Cambiar `ALMACEN = "nube"` en `src/config.py`.

**Importante:** hoy el "usuario" es solo un nombre, sin contraseña. Antes de guardar fotos reales en la nube hay que agregar **inicio de sesión** (Supabase y Firebase lo traen incluido), porque si no, cualquiera que escriba el mismo nombre vería las fotos de otra persona.

## 6. Ideas de funciones

- Inicio de sesión real (correo o Google).
- Marcar un outfit como favorito o como "ya lo usé".
- Filtrar por clima, además de la ocasión.
- Versión para celular: Streamlit ya se adapta a pantallas chicas; revisar que las columnas se vean bien.
- Publicar la app en **Streamlit Community Cloud** (botón *Deploy*) para tener un link público.
