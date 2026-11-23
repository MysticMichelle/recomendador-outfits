"""
reglas.py
---------
Sistema basado en reglas (teoría del color) que califica un outfit.

Cada prenda es un diccionario con al menos:
  "color": {"nombre", "tono", "saturacion", "brillo", "neutro", "fuerte"}
  "patron": "liso", "rayas", ...   ("liso" = sin estampado)
  "mezclilla": True/False          (la mezclilla cuenta como neutro)

Hay dos tipos de outfit:
  · arriba + abajo (+ calzado)
  · completo (vestido, overol) + calzado

Si el usuario elige una OCASIÓN, primero solo entran las prendas etiquetadas
para esa ocasión (filtro) y luego se suma la COLORIMETRÍA de esa ocasión.
"""

from itertools import product


def estampado(p):
    return p.get("patron", "liso") != "liso"


def color_de(p):
    """Color usado por las reglas: la mezclilla se trata como neutro."""
    c = dict(p["color"])
    if p.get("mezclilla"):
        c["neutro"], c["fuerte"], c["nombre"] = True, False, f"{c['nombre']} (mezclilla)"
    return c


def diferencia_tono(t1, t2):
    """Distancia angular entre dos tonos en la rueda de color (0-180°)."""
    d = abs(t1 - t2) % 360
    return min(d, 360 - d)


def calificar_par(c1, c2):
    """Califica qué tan bien combinan dos colores. Regresa (puntos, razones)."""
    if c1["neutro"] and c2["neutro"]:
        return 3, [f"{c1['nombre']} + {c2['nombre']}: dos neutros, combinación segura (+3)"]
    if c1["neutro"] or c2["neutro"]:
        return 3, [f"{c1['nombre']} + {c2['nombre']}: un neutro combina con todo (+3)"]

    d = diferencia_tono(c1["tono"], c2["tono"])
    if d <= 15:
        pts, razon = 2, "mismo tono, monocromático (+2)"
    elif d <= 45:
        pts, razon = 3, "colores análogos, vecinos en la rueda (+3)"
    elif 150 <= d <= 180:
        pts, razon = 3, "colores complementarios, opuestos en la rueda (+3)"
    elif 105 <= d <= 135:
        pts, razon = 1, "colores en triada (+1)"
    else:
        pts, razon = -1, "tonos sin relación armónica (-1)"
    razones = [f"{c1['nombre']} + {c2['nombre']}: {razon}"]

    if c1["fuerte"] and c2["fuerte"] and pts < 3:
        pts -= 2
        razones.append("dos colores muy fuertes juntos (-2)")
    return pts, razones


# ---------------------------------------------------------------------------
# Colorimetría por ocasión: puntos extra según el estilo elegido
# ---------------------------------------------------------------------------
NEUTROS_OSCUROS = ("negro", "azul marino", "gris", "café")
LOGOS = ("gráfico / logo", "animal print")


def colorimetria(prendas, ocasion):
    """Puntos extra (o en contra) según la ocasión. Regresa (puntos, razones)."""
    if ocasion in (None, "casual"):
        return 0, []
    pts, razones = 0, []
    colores = [color_de(p) for p in prendas]
    fuertes = sum(c["fuerte"] for c in colores)
    neutros = sum(c["neutro"] for c in colores)
    logos = sum(p.get("patron") in LOGOS for p in prendas)

    if ocasion == "business_casual":
        if neutros >= len(prendas) - 1:
            pts += 2; razones.append("business casual: casi todo en neutros (+2)")
        if fuertes:
            pts -= 2; razones.append("business casual: evita colores muy intensos (-2)")
        if logos:
            pts -= 2; razones.append("business casual: evita logos o animal print (-2)")
    elif ocasion == "formal":
        oscuros = sum(any(c["nombre"].startswith(n) for n in NEUTROS_OSCUROS) for c in colores)
        if oscuros >= 1:
            pts += 2; razones.append("formal: base en neutros oscuros (+2)")
        if fuertes:
            pts -= 2; razones.append("formal: evita colores muy intensos (-2)")
        if any(estampado(p) and p.get("patron") not in ("rayas", "cuadros") for p in prendas):
            pts -= 2; razones.append("formal: solo rayas o cuadros discretos (-2)")
    elif ocasion == "deportivo":
        if any(c["fuerte"] for c in colores):
            pts += 1; razones.append("deportivo: un color vivo da energía (+1)")
    elif ocasion == "fiesta":
        if any(c["fuerte"] or c["nombre"].startswith(("negro", "vino")) for c in colores):
            pts += 2; razones.append("fiesta: una pieza llamativa o negro/vino (+2)")
        if neutros == len(prendas):
            pts -= 1; razones.append("fiesta: todo neutro se ve apagado (-1)")
    return pts, razones


def escalar(puntos, minimo, maximo):
    calif = 10 * (puntos - minimo) / (maximo - minimo)
    return round(max(0.0, min(10.0, calif)), 1)


def calificar_outfit(arriba, abajo, calzado=None, ocasion=None):
    """Outfit arriba + abajo (+ calzado). Regresa (calificación 0-10, razones)."""
    ca, cb = color_de(arriba), color_de(abajo)
    puntos, razones = calificar_par(ca, cb)
    puntos *= 2  # arriba+abajo es la combinación más importante

    contraste = abs(ca["brillo"] - cb["brillo"])
    if contraste >= 0.35:
        puntos += 1; razones.append(f"buen contraste claro/oscuro ({contraste:.2f}) (+1)")
    elif contraste < 0.1:
        puntos -= 1; razones.append("los dos tienen casi el mismo brillo, se ven planos (-1)")

    if estampado(arriba) and estampado(abajo):
        puntos -= 3; razones.append("dos estampados juntos (-3)")
    elif estampado(arriba) or estampado(abajo):
        lisa = cb if estampado(arriba) else ca
        if lisa["neutro"]:
            puntos += 1; razones.append("estampado equilibrado con prenda lisa neutra (+1)")

    piezas = [arriba, abajo]
    if calzado is not None:
        cz = color_de(calzado)
        if cz["neutro"]:
            puntos += 2; razones.append(f"calzado {cz['nombre']} neutro (+2)")
        else:
            mejor = max(calificar_par(cz, ca)[0], calificar_par(cz, cb)[0])
            puntos += mejor; razones.append(f"calzado {cz['nombre']} con el resto del outfit ({mejor:+d})")
        if estampado(calzado) and (estampado(arriba) or estampado(abajo)):
            puntos -= 2; razones.append("calzado estampado con otra prenda estampada (-2)")
        piezas.append(calzado)

    extra, r = colorimetria(piezas, ocasion)
    puntos += extra; razones += r
    minimo, maximo = (-12, 10) if calzado is not None else (-10, 8)
    if ocasion not in (None, "casual"):
        minimo, maximo = minimo - 4, maximo + 2
    return escalar(puntos, minimo, maximo), razones


def calificar_completo(completo, calzado, ocasion=None):
    """Outfit de una pieza (vestido, overol) + calzado."""
    cc, cz = color_de(completo), color_de(calzado)
    pts, razones = calificar_par(cc, cz)
    puntos = pts * 2
    if estampado(completo) and estampado(calzado):
        puntos -= 3; razones.append("dos estampados juntos (-3)")
    elif estampado(completo) and cz["neutro"]:
        puntos += 1; razones.append("estampado equilibrado con calzado neutro (+1)")
    extra, r = colorimetria([completo, calzado], ocasion)
    puntos += extra; razones += r
    minimo, maximo = -9, 7
    if ocasion not in (None, "casual"):
        minimo, maximo = minimo - 4, maximo + 2
    return escalar(puntos, minimo, maximo), razones


def filtrar_ocasion(prendas, ocasion):
    """PRIMER FILTRO: solo las prendas etiquetadas para la ocasión elegida."""
    if ocasion is None:
        return prendas
    return [p for p in prendas if ocasion in p.get("ocasiones", [])]


def recomendar(prendas, ocasion=None, top=5):
    """
    prendas: lista de diccionarios con "tipo" (superior, inferior, completo, calzado, otro).
    Prueba TODAS las combinaciones posibles y regresa las 'top' mejores.
    """
    prendas = filtrar_ocasion(prendas, ocasion)
    por_tipo = {t: [p for p in prendas if p["tipo"] == t] for t in ("superior", "inferior", "completo", "calzado")}
    calzados = por_tipo["calzado"] or [None]
    resultados = []
    for a, b, c in product(por_tipo["superior"], por_tipo["inferior"], calzados):
        calif, razones = calificar_outfit(a, b, c, ocasion)
        resultados.append({"piezas": [x for x in (a, b, c) if x is not None],
                           "calificacion": calif, "razones": razones})
    for v, c in product(por_tipo["completo"], por_tipo["calzado"]):
        calif, razones = calificar_completo(v, c, ocasion)
        resultados.append({"piezas": [v, c], "calificacion": calif, "razones": razones})
    resultados.sort(key=lambda r: r["calificacion"], reverse=True)
    return resultados[:top]
