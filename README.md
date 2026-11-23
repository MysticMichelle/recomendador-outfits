# 👕 Recomendador de outfits por ocasión con procesamiento de imágenes

Proyecto final de **Sistemas Inteligentes**.

El usuario elige **para qué ocasión se viste** (casual, business casual, formal, deportivo o fiesta) y sube fotos de su ropa. El sistema:

1. **Filtra** las prendas que sirven para esa ocasión.
2. **Reconoce el tipo de prenda** con un modelo supervisado y muestra su **% de seguridad**; si baja de 60 %, avisa.
3. **Obtiene el color** con k-means (no supervisado) y lo nombra con reglas HSV o con un modelo entrenado, el que acierte más.
4. **Califica cada outfit** del 0 al 10 con reglas de teoría del color + la **colorimetría de la ocasión**, y explica por qué.

Incluye un **tablero analítico** con KPIs.

## Problemática y solución

- **Problemática:** elegir qué ponerse toma tiempo, y es difícil saber qué prendas combinan entre sí y cuáles son adecuadas para cada ocasión (por ejemplo, business casual).
- **Solución:** una web app que, a partir de fotos del clóset, recomienda los mejores outfits para la ocasión elegida y explica por qué funcionan.

## Arquitectura

```
fotos ──► 1_etiquetar.py ──► etiquetas/*.csv  (tipo, color, patrón, mezclilla, ocasiones)
                                   │
                                   ▼
                          2_limpieza.py  (homologación regex, duplicados, nulos,
                                   │       características con k-means)
                                   ▼
                          dataset_limpio.csv ──► 3_eda.py ──► reportes/eda/*.png
                                   │
                                   ▼
                          4_modelado.py  (k-means · 6 modelos de tipo · modelo de color · modelo de ocasiones)
                                   │
                                   ▼
                          modelos/modelos.joblib
                                   │
                                   ▼
                          app.py  (Recomendador por ocasión + Tablero de KPIs)
```

## Estructura

| Archivo / carpeta | Contenido |
|---|---|
| `1_etiquetar.py` | App para etiquetar: tipo, color real, patrón, mezclilla y ocasiones |
| `2_limpieza.py` | Limpieza, homologación, duplicados, características, imputación |
| `3_eda.py` | EDA univariado, bivariado y multivariado (incluye qué tanto aciertan las reglas de color) |
| `4_modelado.py` | k-means (codo + silueta), 6 clasificadores de tipo, modelo de color y modelo de ocasiones |
| `app.py` | Web app: recomendador por ocasión + tablero analítico + Mi clóset |
| `src/closet.py` | Mi clóset: guardar prendas por usuario (se borran solas a los 30 días) |
| `.streamlit/config.toml`, `assets/estilo.css` | Tema y estilos visuales de la app |
| `GUIA_DESARROLLO.md` | Guía para quien continúe el diseño y las funciones |
| `src/color_kmeans.py` | **k-means implementado desde cero** + reglas HSV para nombrar el color |
| `src/caracteristicas.py` | Convierte cada foto en 67 características numéricas |
| `src/reglas.py` | Reglas de teoría del color + colorimetría por ocasión |
| `src/config.py` | Rutas, clases, colores, patrones y ocasiones |
| `demo/generar_fotos_demo.py` | Fotos de prueba para verificar que todo corre |
| `datos/fotos/<Persona>/` | Aquí van las fotos (no se suben a GitHub) |

## Etiquetas (por foto)

| Campo | Valores |
|---|---|
| Tipo | 👕 Arriba · 👖 Abajo · 👗 Completo (vestido, overol) · 👟 Calzado · 👜 Otro |
| Color real | negro, blanco, gris, beige, café, azul marino, azul, celeste, turquesa, verde, amarillo, naranja, rojo, vino, rosa, morado, multicolor |
| Patrón | liso, rayas, cuadros, floral, lunares, animal print, gráfico / logo, otro estampado |
| Mezclilla | sí / no (es una tela, no un color) |
| Ocasiones (varias) | Casual / diario · Business casual · Formal / evento · Deportivo · Fiesta / noche |

## Instalación

Requiere **Python 3.13** en Windows (con 3.15, Windows bloquea una librería de pandas).

```bash
py install 3.13
py -3.13 -m pip install -r requirements.txt
```

## Comandos de ejecución (en orden)

```bash
py -3.13 -m streamlit run 1_etiquetar.py   # 1. etiquetar
py -3.13 2_limpieza.py                    # 2. limpieza
py -3.13 3_eda.py                         # 3. EDA
py -3.13 4_modelado.py                    # 4. modelos
py -3.13 -m streamlit run app.py          # 5. web app
```

Para probar sin fotos reales: `py -3.13 demo/generar_fotos_demo.py` (y `--borrar` para quitarlas).

## Modelado

**No supervisado**
- *k-means desde cero* (`src/color_kmeans.py`): agrupa los píxeles de cada foto; el grupo más grande que no es fondo es el color de la prenda.
- *k-means de scikit-learn* sobre todas las fotos: se elige *k* con el método del codo y el coeficiente de silueta.

**Supervisado**
- **Tipo de prenda** (5 clases): se comparan KNN, regresión logística, árbol de decisión, Random Forest, SVM y red neuronal (MLP) con accuracy, precisión, recall y F1 macro, más validación cruzada de 5 particiones. Se usa `class_weight="balanced"` por el desbalance de clases. El mejor modelo se elige por F1 de validación cruzada y da el **% de seguridad**.
- **Color:** se compara la accuracy de las reglas HSV contra un Random Forest entrenado con el color real etiquetado; la app usa el mejor.
- **Ocasiones:** Random Forest multi-etiqueta (una prenda puede servir para varias ocasiones), evaluado con F1 por ocasión.

## Reglas de color

| Regla | Puntos |
|---|---|
| Neutro (negro, blanco, gris, beige, café, azul marino o mezclilla) + cualquier color | +3 |
| Análogos (≤ 45° en la rueda de color) | +3 |
| Complementarios (150°–180°) | +3 |
| Monocromático | +2 |
| Triada (105°–135°) | +1 |
| Sin relación | −1 |
| Dos colores muy fuertes sin armonía | −2 |
| Contraste claro/oscuro entre arriba y abajo (≥ 0.35) | +1 |
| Arriba y abajo con casi el mismo brillo (< 0.1) | −1 |
| Dos estampados juntos | −3 |
| Estampado + prenda lisa neutra | +1 |
| Calzado neutro | +2 |

**Colorimetría por ocasión** (se suma después del filtro):

| Ocasión | Suma | Resta |
|---|---|---|
| Business casual | casi todo en neutros (+2) | colores muy intensos (−2), logos o animal print (−2) |
| Formal / evento | base en neutros oscuros (+2) | colores muy intensos (−2), estampados que no sean rayas o cuadros (−2) |
| Deportivo | un color vivo (+1) | — |
| Fiesta / noche | una pieza llamativa o negro/vino (+2) | todo neutro (−1) |

## Integrantes

- Angel
- Fany
- Leonardo
- Melanie
- Michelle
- Oliver
