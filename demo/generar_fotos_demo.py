"""
Genera fotos FALSAS (dibujos de prendas) para probar todo el flujo
antes de tener sus fotos etiquetadas. Las etiquetas se escriben "sucias"
a propósito (mayúsculas, acentos, sinónimos) para ver la homologación.

    python demo/generar_fotos_demo.py           # crea datos de prueba
    python demo/generar_fotos_demo.py --borrar  # los elimina
"""

import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageDraw  # noqa: E402

from src.config import ETIQUETAS, FOTOS  # noqa: E402

CARPETA = FOTOS / "_Demo"
CSV = ETIQUETAS / "etiquetas__Demo.csv"

if "--borrar" in sys.argv:
    shutil.rmtree(CARPETA, ignore_errors=True)
    CSV.unlink(missing_ok=True)
    print("Datos de demostración eliminados.")
    sys.exit()

random.seed(0)
CARPETA.mkdir(parents=True, exist_ok=True)
ETIQUETAS.mkdir(parents=True, exist_ok=True)

COLORES = [((200, 30, 40), "rojo"), ((245, 245, 245), "Blanca"), ((20, 20, 20), "negro"),
           ((30, 40, 80), "azul marino"), ((40, 150, 60), "verde"), ((130, 40, 170), "morado"),
           ((230, 200, 60), "amarillo"), ((150, 150, 150), "gris"), ((210, 190, 160), "beige"),
           ((225, 165, 175), "rosa"), ((110, 70, 40), "marrón"), ((90, 130, 190), "azul")]
OCASION_POR_TIPO = {"superior": ["casual", "Business", "fiesta"], "inferior": ["casual", "oficina", "formal"],
                    "completo": ["fiesta", "evento formal"], "calzado": ["casual", "deporte", "formal"], "otro": [""]}
SINONIMOS = {
    "superior": ["Playera", "camisa ", "SUDADERA", "Suéter", "arriba"],
    "inferior": ["Pantalón", "falda", "JEANS", " short", "abajo"],
    "calzado": ["Tenis", "zapatos", "Botas", "calzado"],
    "otro": ["Bolsa", "gorra", "accesorio"],
    "completo": ["Vestido", "overol", "jumpsuit"],
}


def dibujar(tipo, color, fondo):
    w, h = random.choice([(300, 400), (400, 300), (320, 320)])
    img = Image.new("RGB", (w, h), fondo)
    d = ImageDraw.Draw(img)
    cx, cy, s = w / 2, h / 2, min(w, h) / 2
    if tipo == "superior":  # torso + mangas
        d.rectangle([cx - .45 * s, cy - .5 * s, cx + .45 * s, cy + .7 * s], fill=color)
        d.polygon([(cx - .45 * s, cy - .5 * s), (cx - .9 * s, cy), (cx - .7 * s, cy + .15 * s),
                   (cx - .45 * s, cy - .1 * s)], fill=color)
        d.polygon([(cx + .45 * s, cy - .5 * s), (cx + .9 * s, cy), (cx + .7 * s, cy + .15 * s),
                   (cx + .45 * s, cy - .1 * s)], fill=color)
    elif tipo == "inferior":  # cintura + dos piernas
        d.rectangle([cx - .4 * s, cy - .8 * s, cx + .4 * s, cy - .5 * s], fill=color)
        d.polygon([(cx - .4 * s, cy - .5 * s), (cx - .02 * s, cy - .5 * s), (cx - .08 * s, cy + .9 * s),
                   (cx - .45 * s, cy + .9 * s)], fill=color)
        d.polygon([(cx + .02 * s, cy - .5 * s), (cx + .4 * s, cy - .5 * s), (cx + .45 * s, cy + .9 * s),
                   (cx + .08 * s, cy + .9 * s)], fill=color)
    elif tipo == "calzado":  # suela + empeine
        d.rectangle([cx - .8 * s, cy + .2 * s, cx + .8 * s, cy + .4 * s], fill=(240, 240, 240))
        d.polygon([(cx - .8 * s, cy + .2 * s), (cx - .6 * s, cy - .4 * s), (cx, cy - .4 * s),
                   (cx + .8 * s, cy + .2 * s)], fill=color)
    elif tipo == "completo":  # vestido: torso + falda ancha
        d.rectangle([cx - .3 * s, cy - .85 * s, cx + .3 * s, cy - .1 * s], fill=color)
        d.polygon([(cx - .3 * s, cy - .1 * s), (cx + .3 * s, cy - .1 * s), (cx + .7 * s, cy + .9 * s),
                   (cx - .7 * s, cy + .9 * s)], fill=color)
    else:  # bolsa con asa
        d.rectangle([cx - .6 * s, cy - .2 * s, cx + .6 * s, cy + .7 * s], fill=color)
        d.arc([cx - .4 * s, cy - .8 * s, cx + .4 * s, cy + .1 * s], 180, 360, fill=color, width=12)
    # ruido para que parezca foto
    px = img.load()
    for _ in range(w * h // 40):
        x, y = random.randrange(w), random.randrange(h)
        r, g, b = px[x, y]
        k = random.randint(-25, 25)
        px[x, y] = (max(0, min(255, r + k)), max(0, min(255, g + k)), max(0, min(255, b + k)))
    return img


filas = ["archivo,persona,tipo,color,patron,mezclilla,ocasiones,fecha"]
n = 0
for tipo, cuantas in [("superior", 45), ("inferior", 40), ("completo", 20), ("calzado", 30), ("otro", 20)]:
    for _ in range(cuantas):
        color, nombre_color = random.choice(COLORES)
        fondo = random.choice([(235, 235, 230), (200, 200, 210), (180, 160, 140)])
        nombre = f"demo_{n:03d}.jpg"
        dibujar(tipo, color, fondo).save(CARPETA / nombre, quality=90)
        ocas = "|".join(random.sample(OCASION_POR_TIPO[tipo], k=min(2, len(OCASION_POR_TIPO[tipo]))))
        patron = random.choice(["", "liso", "rayas", "Logo", "floral"])
        mezcl = random.choice(["0", "1"]) if nombre_color == "azul" else "0"
        filas.append(f"_Demo/{nombre},_demo,{random.choice(SINONIMOS[tipo])},{nombre_color},{patron},{mezcl},{ocas},")
        n += 1
# casos sucios a propósito: duplicado, foto inexistente, etiqueta vacía, descartada
shutil.copy(CARPETA / "demo_000.jpg", CARPETA / "demo_copia.jpg")
filas += ["_Demo/demo_copia.jpg,_demo,playera,rojo,,0,casual,", "_Demo/no_existe.jpg,_demo,falda,negro,,0,casual,",
          "_Demo/demo_001.jpg,_demo,,rojo,,0,casual,", "_Demo/demo_002.jpg,_demo,descartar,,,0,,"]
CSV.write_text("\n".join(filas) + "\n", encoding="utf-8")
print(f"Creadas {n} fotos de prueba en {CARPETA} y etiquetas en {CSV}")
