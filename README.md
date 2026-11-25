# Recomendador de outfits por ocasión con procesamiento de imágenes

Proyecto final de **Sistemas Inteligentes** · UNAM FES Acatlán · MAC · Grupo 1704 · 2027-I

Repositorio: https://github.com/MysticMichelle/recomendador-outfits

El usuario elige **para qué ocasión se viste** (casual, business casual, formal, deportivo o fiesta) y sube fotos de su ropa. El sistema:

1. **Filtra** las prendas que sirven para esa ocasión.
2. **Reconoce el tipo de prenda y sus ocasiones** con una **red neuronal convolucional preentrenada (MobileNetV2)** como extractor de características + un Random Forest.
3. **Obtiene el color** con k-means (no supervisado) y lo nombra con reglas HSV o con un modelo entrenado, el que acierte más.
4. **Califica cada outfit** del 0 al 10 con reglas de teoría del color + la **colorimetría de la ocasión**, y explica por qué.

El usuario puede corregir lo detectado y guardar sus prendas en **Mi clóset** (30 días).

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
                          5_cnn_experimento.py  (MobileNetV2 → mismos 6 modelos; se adopta si mejora el F1)
                                   │
                                   ▼
                          modelos/modelos.joblib
                                   │
                                   ▼
                          app.py  (Recomendador por ocasión + Mi clóset)
```

## Estructura

| Archivo / carpeta | Contenido |
|---|---|
| `1_etiquetar.py` | App para etiquetar: tipo, color real, patrón, mezclilla y ocasiones |
| `2_limpieza.py` | Limpieza, homologación, duplicados, características, imputación |
| `3_eda.py` | EDA univariado, bivariado y multivariado (incluye qué tanto aciertan las reglas de color) |
| `4_modelado.py` | k-means (codo + silueta), 6 clasificadores de tipo, modelo de color y modelo de ocasiones |
| `5_cnn_experimento.py` | Características de MobileNetV2 (transfer learning) y comparación con las manuales |
| `app.py` | Web app: recomendador por ocasión + Mi clóset |
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
| Tipo | Arriba · Abajo · Completo (vestido, overol) · Calzado · Otro |
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
py -3.13 5_cnn_experimento.py             # 5. CNN preentrenada (requiere las fotos)
py -3.13 -m streamlit run app.py          # 6. web app
```

> El repositorio incluye los modelos entrenados (`modelos/modelos.joblib`, `modelos/modelos_cnn.joblib`) y MobileNetV2 en formato ONNX (`modelos/mobilenetv2-12.onnx`), así que la app se puede abrir directamente.
> `4_modelado.py` no necesita las fotos (usa `datos/dataset_limpio.csv`); `5_cnn_experimento.py` sí, porque pasa cada foto por la red.

Para probar sin fotos reales: `py -3.13 demo/generar_fotos_demo.py` (y `--borrar` para quitarlas).

## Modelado

**No supervisado**
- *k-means desde cero* (`src/color_kmeans.py`): agrupa los píxeles de cada foto; el grupo más grande que no es fondo es el color de la prenda.
- *k-means de scikit-learn* sobre todas las fotos: se elige *k* con el método del codo y el coeficiente de silueta.

**Supervisado**
- **Tipo de prenda** (5 clases): se comparan KNN, regresión logística, árbol de decisión, Random Forest, SVM y red neuronal (MLP) con accuracy, precisión, recall y F1 macro, más validación cruzada de 5 particiones. Se usa `class_weight="balanced"` por el desbalance de clases. El mejor modelo se elige por F1 de validación cruzada.
- **CNN preentrenada (transfer learning):** cada foto pasa por MobileNetV2 (entrenada con ImageNet, formato ONNX con `onnxruntime`, sin PyTorch ni TensorFlow) y sus 1 000 salidas se usan como características para los mismos 6 clasificadores, con la misma partición y validación cruzada. Si supera al modelo manual en F1 de validación cruzada, la app la usa para el tipo y las ocasiones.
- **Color:** se compara la accuracy de las reglas HSV contra un Random Forest entrenado con el color real etiquetado; la app usa el mejor.
- **Ocasiones:** un clasificador binario por ocasión (una prenda puede servir para varias); se comparan Random Forest y regresión logística; para cada ocasión se ajusta el umbral de decisión que maximiza F0.5 (prioriza la precisión, porque la ocasión se usa como filtro) y se elige el modelo con mejor F0.5 promedio.

## Resultados (1 043 fotos)

| Paso | Resultado |
|---|---|
| Limpieza | 1 175 registros leídos → 128 sin etiqueta válida/descartados, 4 duplicados → **1 043 fotos**, 0 nulos |
| Tipos | superior 588 · inferior 239 · completo 116 · calzado 68 · otro 32 |
| Prendas neutras | 71.8 % |
| k-means (todas las fotos) | mejor *k* = 3, silueta 0.099: los grupos **no** corresponden a los tipos de prenda |
| Tipo de prenda (manual, 67 características) | Random Forest: accuracy 0.732, F1 macro 0.628 (validación cruzada 0.630 ± 0.075) |
| **Tipo de prenda (CNN, 1 000 características)** | **Random Forest: accuracy 0.818, F1 macro 0.781 (validación cruzada 0.875 ± 0.059)** ← usado en la app |
| Color | reglas HSV 0.512 → **Random Forest 0.607** |
| Ocasiones (CNN) | **Random Forest** con umbral por ocasión: precisión promedio 0.629, recall 0.505, F1 0.545 (manual: 0.673 / 0.353 / 0.432) |

| Modelo de tipo (F1 validación cruzada) | Manual (67) | CNN (1 000) | CNN + manual (1 067) |
|---|---|---|---|
| Random Forest | 0.630 | **0.875** (seleccionado) | 0.873 |
| SVM (RBF) | 0.617 | 0.864 | 0.862 |
| Red neuronal (MLP) | 0.579 | 0.862 | 0.863 |
| Regresión logística | 0.501 | 0.855 | 0.856 |
| KNN (k = 5) | 0.525 | 0.802 | 0.806 |
| Árbol de decisión | 0.424 | 0.671 | 0.656 |

Las gráficas están en `reportes/eda/` y `reportes/modelado/`.

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

- Deciga Torres Melanie
- Domínguez Lira Estefani Michalle
- Escobar Sánchez Leonardo
- Heredia Tamayo Oliver de Yahve
- Hernández Peña Angel Adrian
- Moctezuma Isidro Michelle

Profesora: Mirian Ángeles Uribe
