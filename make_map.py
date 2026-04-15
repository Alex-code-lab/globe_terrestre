#!/usr/bin/env python3
"""
Flat Map 2D STL Generator — Puzzle d'encastrement
Converts BlankMap_World_simple3.svg to 3D-printable STL files.

Géométrie :
  map_base.stl           — plaque rectangulaire (océan) pleine avec creux
                           en forme de continents (non traversants) + rebord périphérique
  map_base_left/right.stl— (option) base découpée en 2 moitiés (coupe droite)
  continents/<name>.stl  — pièces individuelles (corps + collerette + bouton de prise)

Principe d'encastrement (même que le globe) :
  - Le corps du continent s'insère dans le creux (jeu INSERT_CLEARANCE).
  - La collerette (LIP_W mm plus large que le creux) repose sur la surface
    océan et retient la pièce, qui ne peut pas tomber.
  - Pour retirer : on soulève par la collerette.

Même découpage que make_globe.py :
  - même SVG source, même seuil LAND_THRESHOLD, mêmes graines de nommage
  - même option de coupures géographiques (Suez, Panama, Oural…)
  - même filtrage des micro-îles (MIN_CONTINENT_QUADS)
"""

import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from stl import mesh

# ── Configuration ──────────────────────────────────────────────────────────────
SVG_FILE   = Path("/Users/souchaud/Documents/Impression 3D/globe_terrestre/BlankMap_World_simple4.svg")
OUTPUT_DIR = Path.home() / "Desktop/flat_map"

MAP_MAX_LEN = 330.0          # mm — longueur max demandée
MAP_W  = MAP_MAX_LEN         # mm — largeur totale (axe X)
MAP_H  = MAP_W / 2.0         # mm — hauteur totale  (rapport 2:1, projection équirectangulaire)

GRID_X =  900    # quads en X (résolution du maillage STL, allégée pour slicer)
GRID_Y =  450    # quads en Y
IMG_W  = 4000    # px — résolution de rasterisation (multiple de GRID pour précision)
IMG_H  = 2000    # px

LAND_THRESHOLD      = 230   # seuil gris : < seuil → terre  (idem make_globe.py)
MIN_CONTINENT_QUADS =  40   # ignorer les composantes < N quads

BASE_THICK       = 5.0   # mm — épaisseur de la plaque de base
POCKET_DEPTH     = 3.0   # mm — profondeur des creux continents (doit être < BASE_THICK)
INSERT_CLEARANCE = 0.3   # mm — jeu entre le corps du continent et son logement
                          #      0   = ajustement exact
                          #      > 0 = jeu (plus facile à insérer/retirer)
LIP_W            = 1.2   # mm — largeur de la collerette (dépasse du creux)
LIP_DEPTH        = 0.8   # mm — épaisseur de la collerette
LIP_SMOOTH_MM    = 0.30  # mm — lissage du contour de jupe (supprime dents/pics)
BORDER_W_MM      = 2.0   # mm — largeur du rebord périphérique (hors jonction)
BORDER_H_MM      = 2.0   # mm — hauteur du rebord périphérique
BUTTON_R_MM      = 3.6   # mm — rayon du bouton de préhension des continents (2× plus large)
BUTTON_H_MM      = 10.0  # mm — hauteur du bouton au-dessus de la face visible
BUTTON_SEGMENTS  = 24    # segments du cylindre (plus = plus rond)
BUTTON_ANCHOR_MM = 0.4   # mm — ancrage du bouton dans la pièce (vers le bas)

# Continents avec bouton de préhension (autres îles n'en ont pas)
BUTTON_CONTINENTS = {"north_america", "south_america", "africa", "europe", "asia", "australia", "antarctica"}

ENABLE_GEOGRAPHIC_CUTS = True    # Suez, Panama, Oural… (nécessaire pour séparer Amériques)
ENABLE_RED_CUTS        = False   # traits rouges dans le SVG source
MIRROR_X               = True    # True = miroir horizontal (gauche/droite)
MIRROR_Y               = False   # True = miroir vertical (haut/bas)

# Découpe plateau en 2 pièces (coupe droite au milieu, sans attache spéciale).
SPLIT_BASE_IN_TWO      = True   # True = export map_base_left/right.stl

# Graines de nommage (identiques à make_globe.py)
CONTINENT_SEEDS = [
    ( 40.0, -100.0, "north_america"),
    (-15.0,  -55.0, "south_america"),
    (  5.0,   20.0, "africa"),
    ( 50.0,   15.0, "europe"),
    ( 50.0,  100.0, "asia"),
    (-25.0,  135.0, "australia"),
    (-80.0,    0.0, "antarctica"),
    ( 72.0,  -42.0, "greenland"),
]
# ──────────────────────────────────────────────────────────────────────────────

if not (0.0 < POCKET_DEPTH < BASE_THICK):
    raise ValueError("POCKET_DEPTH doit être > 0 et strictement < BASE_THICK.")
if not (0.0 <= LIP_DEPTH < POCKET_DEPTH):
    raise ValueError("LIP_DEPTH doit être >= 0 et strictement < POCKET_DEPTH.")
if LIP_SMOOTH_MM < 0.0:
    raise ValueError("LIP_SMOOTH_MM doit être >= 0.")
if BORDER_W_MM < 0.0 or BORDER_H_MM < 0.0:
    raise ValueError("BORDER_W_MM/BORDER_H_MM doivent être >= 0.")
if BUTTON_R_MM < 0.0 or BUTTON_H_MM < 0.0:
    raise ValueError("BUTTON_R_MM/BUTTON_H_MM doivent être >= 0.")
if BUTTON_SEGMENTS < 8:
    raise ValueError("BUTTON_SEGMENTS doit être >= 8.")
if BUTTON_ANCHOR_MM < 0.0:
    raise ValueError("BUTTON_ANCHOR_MM doit être >= 0.")


# ── Rasterisation SVG (identique à make_globe.py) ─────────────────────────────

def _rgba_to_gray_and_red(img_rgba: Image.Image):
    white = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    white.paste(img_rgba, mask=img_rgba.split()[3])
    rgb = np.array(white.convert("RGB"))
    red_mask = (
        (rgb[:, :, 0] > 100) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80)
        & (rgb[:, :, 0] > rgb[:, :, 1] * 2)
        & (rgb[:, :, 0] > rgb[:, :, 2] * 2)
    )
    rgb[red_mask] = 255
    return np.array(Image.fromarray(rgb).convert("L")), red_mask


def rasterize_svg(svg_path: Path, width: int, height: int):
    try:
        import cairosvg
        data = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
        return _rgba_to_gray_and_red(Image.open(io.BytesIO(data)).convert("RGBA"))
    except Exception as e:
        print(f"  cairosvg failed ({e}), trying rsvg-convert…")

    tmp = Path("/tmp/_flat_map.png")
    try:
        r = subprocess.run(
            ["rsvg-convert", "-w", str(width), "-h", str(height), "-o", str(tmp), str(svg_path)],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0:
            return _rgba_to_gray_and_red(Image.open(tmp).convert("RGBA"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        r = subprocess.run(
            ["inkscape", "--export-type=png",
             f"--export-filename={tmp}",
             f"--export-width={width}", f"--export-height={height}", str(svg_path)],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0:
            return _rgba_to_gray_and_red(Image.open(tmp).convert("RGBA"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise RuntimeError(
        "Impossible de rasteriser le SVG.\n"
        "Installez cairosvg (pip install cairosvg) ou librsvg (brew install librsvg)."
    )


# ── Coupures géographiques (identiques à make_globe.py) ───────────────────────

def _draw_cut(img, lat0, lon0, lat1, lon1, width_px=4):
    h, w = img.shape
    x0 = int((lon0 + 180) / 360 * w);  y0 = int((90 - lat0) / 180 * h)
    x1 = int((lon1 + 180) / 360 * w);  y1 = int((90 - lat1) / 180 * h)
    n  = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.round(np.linspace(x0, x1, n)).astype(int)
    ys = np.round(np.linspace(y0, y1, n)).astype(int)
    for dy in range(-width_px, width_px + 1):
        for dx in range(-width_px, width_px + 1):
            img[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] = 255


def _apply_geographic_cuts(land_map):
    img = land_map.copy()
    _draw_cut(img,  9.5, -80.0,  8.5, -77.0, 3)   # Panama
    _draw_cut(img, 32.0,  32.5, 28.5,  32.5, 3)   # Suez
    _draw_cut(img, 36.2,  -6.2, 35.5,  -4.6, 3)   # Gibraltar
    _draw_cut(img, 68.0,  60.0, 51.0,  59.0, 4)   # Oural N
    _draw_cut(img, 51.0,  59.0, 47.0,  51.5, 4)   # Oural S
    _draw_cut(img, 43.0,  48.0, 41.5,  41.0, 4)   # Caucase
    _draw_cut(img, 41.4,  29.2, 40.2,  26.0, 3)   # Bosphore
    return img


# ── Masques quad (image → grille STL) ─────────────────────────────────────────

def _img_to_quad(img_mask):
    """
    Downsample une image booléenne (IMG_H, IMG_W) en masque de quads (GRID_Y, GRID_X).
    Un quad est True si la majorité de ses 4 coins image sont True.
    Même logique que _land_flag() de make_globe.py.
    """
    img_h, img_w = img_mask.shape
    i_img = np.round(np.linspace(0, img_h - 1, GRID_Y + 1)).astype(int)
    j_img = np.round(np.linspace(0, img_w - 1, GRID_X + 1)).astype(int)
    flag  = img_mask[np.ix_(i_img, j_img)].astype(float)
    quad = ((flag[:GRID_Y, :GRID_X] + flag[1:, :GRID_X] +
             flag[:GRID_Y, 1:]      + flag[1:, 1:]) / 4.0) > 0.5
    if MIRROR_X:
        quad = quad[:, ::-1]
    if MIRROR_Y:
        quad = quad[::-1, :]
    return quad


def _fix_diagonal_contacts(mask):
    """
    Supprime les ambiguïtés 2x2 de type diagonale (X) qui peuvent générer
    des arêtes non-manifold sur les parois verticales.
    """
    m = mask.copy()
    a = m[:-1, :-1]
    b = m[:-1, 1:]
    c = m[1:, :-1]
    d = m[1:, 1:]

    diag_ad = a & d & ~b & ~c
    diag_bc = b & c & ~a & ~d

    # Ferme le carré 2x2 pour éviter les jonctions diagonales ambiguës.
    m[:-1, 1:][diag_ad] = True
    m[1:, :-1][diag_ad] = True
    m[:-1, :-1][diag_bc] = True
    m[1:, 1:][diag_bc] = True
    return m


# ── Segmentation continents ────────────────────────────────────────────────────

def _name_component(mask, img_h, img_w):
    """
    Nomme un composant via deux passes :
    - Passe 1 (directe) : la graine tombe dans le composant (rayon 10 px ≈ 0.9°).
      Fiable et sans risque de débordement vers un continent voisin.
    - Passe 2 (centroïde) : fallback — renvoie la graine la plus proche du centroïde
      du composant. Nécessaire quand la graine tombe dans un gap SVG.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    cy, cx = float(rows.mean()), float(cols.mean())

    # Passe 1 : graine directement dans le composant (petit rayon, sans ambiguïté)
    r_direct = 10
    for lat, lon, name in CONTINENT_SEEDS:
        ci = int(np.clip((90 - lat) / 180 * img_h, 0, img_h - 2))
        cj = int(np.clip((lon + 180) / 360 * img_w, 0, img_w - 2))
        if mask[max(0, ci - r_direct):ci + r_direct + 1,
                max(0, cj - r_direct):cj + r_direct + 1].any():
            return name

    # Passe 2 : centroïde le plus proche d'une graine (pour fragments sans graine directe)
    best_name, best_dist = None, float('inf')
    for lat, lon, name in CONTINENT_SEEDS:
        ci = (90 - lat) / 180 * img_h
        cj = (lon + 180) / 360 * img_w
        d = (cy - ci) ** 2 + (cx - cj) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def label_continents(land_map, red_mask=None):
    """
    Segmente la carte en continents/îles nommés.
    Retourne [(name, quad_mask_body, quad_mask_exact, quad_mask_lip), ...].
      quad_mask_body  : masque quad érodé de INSERT_CLEARANCE  (corps de l'insert)
      quad_mask_exact : masque quad sans jeu (forme du creux dans la base)
    """
    from scipy.ndimage import (
        label as ndlabel, binary_opening,
        binary_erosion, binary_dilation, binary_closing,
        gaussian_filter,
    )

    img = land_map.copy()
    if ENABLE_RED_CUTS and red_mask is not None:
        img[red_mask] = 255
    if ENABLE_GEOGRAPHIC_CUTS:
        img = _apply_geographic_cuts(img)

    land_bin = (img < LAND_THRESHOLD)
    land_bin = binary_opening(land_bin, structure=np.ones((3, 3)))

    # Jeu : INSERT_CLEARANCE → érosion en pixels image
    px_per_mm = IMG_W / MAP_W
    n_clr = max(1, round(abs(INSERT_CLEARANCE) * px_per_mm))
    if INSERT_CLEARANCE > 0:
        land_insert = binary_erosion(land_bin,  iterations=n_clr)
    elif INSERT_CLEARANCE < 0:
        land_insert = binary_dilation(land_bin, iterations=n_clr)
    else:
        land_insert = land_bin.copy()

    # Segmentation sur le masque "insert" (avec jeu)
    labeled, n_comp = ndlabel(land_insert, structure=np.ones((3, 3)))
    img_h, img_w = land_bin.shape

    # Largeur/lissage de jupe dans la grille STL (plus stable pour l'impression)
    quad_mm = MAP_W / GRID_X
    n_lip_q = max(1, round(LIP_W / quad_mm))
    sigma_lip_q = max(0.8, LIP_SMOOTH_MM / quad_mm) if LIP_SMOOTH_MM > 0 else 0.0

    # Pré-calcul des tailles de composantes (O(n) unique) pour court-circuiter
    # les opérations scipy coûteuses sur les milliers de micro-îles inférieures au seuil.
    # ~20 pixels/quad en moyenne (4000/900 × 2000/450) → seuil très conservateur = ×4.
    comp_sizes = np.bincount(labeled.ravel())
    min_px = MIN_CONTINENT_QUADS * 4

    used_names: dict = {}
    components = []
    for cid in range(1, n_comp + 1):
        if comp_sizes[cid] < min_px:   # filtre rapide avant de créer mask_insert
            continue

        mask_insert = labeled == cid

        # Passage en quads pour filtrer les micro-îles
        lq_body = _fix_diagonal_contacts(_img_to_quad(mask_insert))
        if lq_body.sum() < MIN_CONTINENT_QUADS:
            continue

        name = _name_component(mask_insert, img_h, img_w)
        if name is None:
            name = f"island_{cid:03d}"
        elif name in used_names:
            used_names[name] += 1
            name = f"{name}_{used_names[name]}"
        else:
            used_names[name] = 1

        # Masque exact (sans jeu) → forme du creux dans la base
        # On trouve la composante connexe correspondante dans land_bin
        # en cherchant l'overlap avec mask_insert dilaté
        overlap = land_bin & binary_dilation(mask_insert, iterations=n_clr + 2)
        exact_labeled, _ = ndlabel(overlap, structure=np.ones((3, 3)))
        seed_ci, seed_cj = np.where(mask_insert)
        if len(seed_ci) == 0:
            lq_exact = lq_body
        else:
            mid = len(seed_ci) // 2
            exact_id = exact_labeled[seed_ci[mid], seed_cj[mid]]
            if exact_id == 0:
                lq_exact = lq_body
            else:
                lq_exact = _fix_diagonal_contacts(_img_to_quad(exact_labeled == exact_id))

        components.append((name, lq_body, lq_exact))

    # Occupation totale des creux (tous continents confondus)
    occupied_exact = np.zeros((GRID_Y, GRID_X), dtype=bool)
    for _, _, lq_exact in components:
        occupied_exact |= lq_exact

    results = []
    for name, lq_body, lq_exact in components:
        # Collerette/jupe : construite en quads puis lissée pour éviter
        # les petites dents/pics visibles sur le débord.
        lq_lip = binary_dilation(lq_body, iterations=n_lip_q)
        if sigma_lip_q > 0:
            lq_lip = gaussian_filter(lq_lip.astype(np.float32), sigma=sigma_lip_q) > 0.45
        lq_lip = binary_closing(lq_lip, structure=np.ones((3, 3)), iterations=1)
        lq_lip = binary_opening(lq_lip, structure=np.ones((3, 3)), iterations=1)
        # Évite des petits îlots de jupe détachés du contour du continent.
        lq_lip &= binary_dilation(lq_body, iterations=n_lip_q + 1)

        # Bloque la jupe uniquement sur les emplacements des AUTRES continents.
        # (Ne pas bloquer sur son propre creux, sinon une démarcation/gorge apparaît.)
        occupied_other = occupied_exact & ~lq_exact
        lq_lip[occupied_other & ~lq_body] = False
        lq_lip |= lq_body
        lq_lip = _fix_diagonal_contacts(lq_lip)

        results.append((name, lq_body, lq_exact, lq_lip))

    # ── Renommage : le fragment le plus grand de chaque continent prend le nom primaire ──
    # Exemple : si "australia_3" est le plus grand, il devient "australia",
    # et les petits fragments sont renumérotés à partir de _2.
    known_bases = {n for _, _, n in CONTINENT_SEEDS}
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for idx, (name, lq_body, lq_exact, lq_lip) in enumerate(results):
        parts = name.rsplit('_', 1)
        base = parts[0] if len(parts) > 1 and parts[1].isdigit() else name
        if base in known_bases:
            groups[base].append((lq_body.sum(), idx))
    rename_map: dict = {}
    for base, frags in groups.items():
        frags.sort(reverse=True, key=lambda x: x[0])   # plus grand en premier
        for rank, (_, idx) in enumerate(frags):
            old = results[idx][0]
            rename_map[old] = base if rank == 0 else f"{base}_{rank + 1}"
    results = [(rename_map.get(n, n), a, b, c) for n, a, b, c in results]

    for name, lq_body, lq_exact, lq_lip in results:
        print(f"    {name:25s}: {lq_body.sum():>7,} quads (corps)  "
              f"{lq_exact.sum():>7,} (creux)  {lq_lip.sum():>7,} (collerette)")

    return results


# ── Grilles de vertices ───────────────────────────────────────────────────────

def _make_verts(z_val):
    """
    Retourne un tableau (GRID_Y+1, GRID_X+1, 3) de vertices plats à hauteur z_val.
    z_val peut être un scalaire ou un tableau (GRID_Y+1, GRID_X+1).
    """
    xs = np.linspace(0.0, MAP_W, GRID_X + 1)   # (GRID_X+1,)
    ys = np.linspace(0.0, MAP_H, GRID_Y + 1)   # (GRID_Y+1,)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")  # (GRID_Y+1, GRID_X+1)
    if np.ndim(z_val) == 0:
        ZZ = np.full_like(XX, float(z_val))
    else:
        ZZ = z_val
    return np.stack([XX, YY, ZZ], axis=-1).astype(np.float32)


def _split_footprints():
    """
    Construit 2 empreintes complémentaires (gauche/droite) avec une
    coupe droite verticale au milieu, sans languettes ni verrous.
    """
    left = np.zeros((GRID_Y, GRID_X), dtype=bool)
    right = np.zeros((GRID_Y, GRID_X), dtype=bool)

    j_mid = GRID_X // 2
    left[:, :j_mid] = True
    right[:, j_mid:] = True

    return left, right


def _outer_border_mask(footprint_mask, n_q):
    """
    Bande périphérique basée sur le contour du rectangle global de la carte.
    Avec une base coupée en 2, cela exclut naturellement la jonction centrale.
    """
    if n_q <= 0:
        return np.zeros_like(footprint_mask, dtype=bool)

    band = np.zeros_like(footprint_mask, dtype=bool)
    band[:n_q, :] = True
    band[-n_q:, :] = True
    band[:, :n_q] = True
    band[:, -n_q:] = True
    return footprint_mask & band



# ── Primitives triangles ──────────────────────────────────────────────────────

def _top_face(mask, V):
    """Faces supérieures (+Z) pour les quads True du masque."""
    si, sj = np.where(mask)
    t1 = np.stack([V[si, sj],   V[si, sj+1], V[si+1, sj+1]], axis=1)
    t2 = np.stack([V[si, sj],   V[si+1, sj+1], V[si+1, sj]], axis=1)
    return np.concatenate([t1, t2], axis=0)


def _bot_face(mask, V):
    """Faces inférieures (-Z) pour les quads True du masque."""
    si, sj = np.where(mask)
    t1 = np.stack([V[si, sj],   V[si+1, sj+1], V[si, sj+1]], axis=1)
    t2 = np.stack([V[si, sj],   V[si+1, sj],   V[si+1, sj+1]], axis=1)
    return np.concatenate([t1, t2], axis=0)


def _walls_outward(mask, Va, Vb):
    """
    Parois verticales autour du périmètre du masque, normales vers l'extérieur.
    Va = vertices "haut", Vb = vertices "bas".
    """
    tris = []

    # Nord (normal -Y)
    oce = np.zeros_like(mask); oce[0, :] = True; oce[1:, :] = ~mask[:-1, :]
    si, sj = np.where(mask & oce)
    if len(si):
        tris += [np.stack([Va[si, sj],   Vb[si, sj+1], Va[si, sj+1]], axis=1),
                 np.stack([Va[si, sj],   Vb[si, sj],   Vb[si, sj+1]], axis=1)]

    # Sud (normal +Y)
    oce = np.zeros_like(mask); oce[-1, :] = True; oce[:-1, :] = ~mask[1:, :]
    si, sj = np.where(mask & oce)
    if len(si):
        tris += [np.stack([Va[si+1, sj],   Va[si+1, sj+1], Vb[si+1, sj+1]], axis=1),
                 np.stack([Va[si+1, sj],   Vb[si+1, sj+1], Vb[si+1, sj]  ], axis=1)]

    # Ouest (normal -X)
    oce = np.zeros_like(mask); oce[:, 0] = True; oce[:, 1:] = ~mask[:, :-1]
    si, sj = np.where(mask & oce)
    if len(si):
        tris += [np.stack([Va[si, sj],   Va[si+1, sj], Vb[si+1, sj]], axis=1),
                 np.stack([Va[si, sj],   Vb[si+1, sj], Vb[si, sj]  ], axis=1)]

    # Est (normal +X)
    oce = np.zeros_like(mask); oce[:, -1] = True; oce[:, :-1] = ~mask[:, 1:]
    si, sj = np.where(mask & oce)
    if len(si):
        tris += [np.stack([Va[si+1, sj+1], Va[si, sj+1],   Vb[si, sj+1]  ], axis=1),
                 np.stack([Va[si+1, sj+1], Vb[si, sj+1],   Vb[si+1, sj+1]], axis=1)]

    return np.concatenate(tris, axis=0).astype(np.float32) if tris else np.zeros((0, 3, 3), np.float32)


def _walls_from_to(mask_from, mask_to, Va, Vb):
    """
    Parois sur l'interface mask_from -> mask_to, normales sortantes de mask_from.
    Utile pour construire uniquement les parois océan↔creux (sans bords externes).
    """
    tris = []

    # Nord (normal -Y)
    nei = np.zeros_like(mask_from); nei[1:, :] = mask_to[:-1, :]
    si, sj = np.where(mask_from & nei)
    if len(si):
        tris += [np.stack([Va[si, sj],   Vb[si, sj+1], Va[si, sj+1]], axis=1),
                 np.stack([Va[si, sj],   Vb[si, sj],   Vb[si, sj+1]], axis=1)]

    # Sud (normal +Y)
    nei = np.zeros_like(mask_from); nei[:-1, :] = mask_to[1:, :]
    si, sj = np.where(mask_from & nei)
    if len(si):
        tris += [np.stack([Va[si+1, sj],   Va[si+1, sj+1], Vb[si+1, sj+1]], axis=1),
                 np.stack([Va[si+1, sj],   Vb[si+1, sj+1], Vb[si+1, sj]  ], axis=1)]

    # Ouest (normal -X)
    nei = np.zeros_like(mask_from); nei[:, 1:] = mask_to[:, :-1]
    si, sj = np.where(mask_from & nei)
    if len(si):
        tris += [np.stack([Va[si, sj],   Va[si+1, sj], Vb[si+1, sj]], axis=1),
                 np.stack([Va[si, sj],   Vb[si+1, sj], Vb[si, sj]  ], axis=1)]

    # Est (normal +X)
    nei = np.zeros_like(mask_from); nei[:, :-1] = mask_to[:, 1:]
    si, sj = np.where(mask_from & nei)
    if len(si):
        tris += [np.stack([Va[si+1, sj+1], Va[si, sj+1],   Vb[si, sj+1]  ], axis=1),
                 np.stack([Va[si+1, sj+1], Vb[si, sj+1],   Vb[si+1, sj+1]], axis=1)]

    return np.concatenate(tris, axis=0).astype(np.float32) if tris else np.zeros((0, 3, 3), np.float32)


def _walls_inward(mask, Va, Vb):
    """Parois verticales avec normales vers l'intérieur (renversement du winding)."""
    w = _walls_outward(mask, Va, Vb)
    return w[:, [0, 2, 1], :]   # swap v1 ↔ v2


def _outer_walls(Va, Vb):
    """
    Parois des 4 bords extérieurs du rectangle [0,MAP_W] x [0,MAP_H].
    Normales vers l'extérieur de la plaque.
    """
    tris = []
    sj = np.arange(GRID_X)

    # Nord (y=0, normal -Y)
    tris += [np.stack([Va[0, sj],   Vb[0, sj+1], Va[0, sj+1]], axis=1),
             np.stack([Va[0, sj],   Vb[0, sj],   Vb[0, sj+1]], axis=1)]

    # Sud (y=MAP_H, normal +Y)
    tris += [np.stack([Va[GRID_Y, sj],   Va[GRID_Y, sj+1], Vb[GRID_Y, sj+1]], axis=1),
             np.stack([Va[GRID_Y, sj],   Vb[GRID_Y, sj+1], Vb[GRID_Y, sj]  ], axis=1)]

    si = np.arange(GRID_Y)

    # Ouest (x=0, normal -X)
    tris += [np.stack([Va[si, 0],   Va[si+1, 0], Vb[si+1, 0]], axis=1),
             np.stack([Va[si, 0],   Vb[si+1, 0], Vb[si, 0]  ], axis=1)]

    # Est (x=MAP_W, normal +X)
    tris += [np.stack([Va[si+1, GRID_X], Va[si, GRID_X],   Vb[si, GRID_X]  ], axis=1),
             np.stack([Va[si+1, GRID_X], Vb[si, GRID_X],   Vb[si+1, GRID_X]], axis=1)]

    return np.concatenate(tris, axis=0).astype(np.float32)


# ── Construction des STL ───────────────────────────────────────────────────────

def build_base_plate(continent_data, footprint_mask=None):
    """
    Plaque de base : empreinte 2D + creux (non traversants) continents.

    Surfaces :
      1. Face supérieure océan (z=0)          : quads hors continents (+Z)
      2. Rebord périphérique (z=+BORDER_H_MM) : bande externe continue
      3. Fond des creux (z=-POCKET_DEPTH)     : quads continents (+Z), hors zone rebord
      4. Face inférieure (z=-BASE_THICK)      : empreinte complète (-Z)
      5. Parois extérieures                   : contour de l'empreinte
      6. Parois des creux                     : périmètre continents
      7. Parois du rebord                     : montée locale de z=0 vers z=+BORDER_H_MM
    """
    if footprint_mask is None:
        footprint_mask = np.ones((GRID_Y, GRID_X), dtype=bool)

    V_top = _make_verts(0.0)
    V_border = _make_verts(BORDER_H_MM)
    V_pocket = _make_verts(-POCKET_DEPTH)
    V_bot = _make_verts(-BASE_THICK)

    # Union de tous les masques exacts (forme des creux)
    lq_all = np.zeros((GRID_Y, GRID_X), dtype=bool)
    for entry in continent_data:
        lq_all |= entry[2]

    border_q = max(1, round(BORDER_W_MM / (MAP_W / GRID_X))) if (BORDER_W_MM > 0 and BORDER_H_MM > 0) else 0
    border = _outer_border_mask(footprint_mask, border_q)

    # Rebord prioritaire : on retire les creux dans la bande périphérique
    # pour garantir un rebord continu même si un continent touche le bord.
    pockets = footprint_mask & lq_all & ~border
    top_support = footprint_mask & ~pockets
    ocean_flat = top_support & ~border

    tris = [
        _top_face(ocean_flat, V_top),
        _top_face(border, V_border),
        _top_face(pockets, V_pocket),
        _bot_face(footprint_mask, V_bot),
        _walls_outward(footprint_mask, V_top, V_bot),
        _walls_inward(pockets, V_top, V_pocket),                 # parois complètes des creux
        _walls_outward(border, V_border, V_top),           # parois du rebord
    ]
    return np.concatenate(tris, axis=0).astype(np.float32)


def build_continent_insert(lq_body, lq_lip, add_button=False):
    """
    Pièce continent à encastrer, avec collerette de retenue.

    Niveaux Z :
      z = 0          — face visible (top)
      z = -LIP_DEPTH — dessous de la collerette
      z = -POCKET_DEPTH — fond du corps (repose sur le fond du creux)

    Surfaces :
      1. Face sup (z=0)        : collerette + corps (+Z)  ← sert aussi de base du bouton
      2. Dessous collerette    : anneau (collerette − corps) à z=-LIP_DEPTH (-Z)
      3. Fond du corps         : corps à z=-POCKET_DEPTH (-Z)
      4. Parois ext collerette : de z=0 à z=-LIP_DEPTH  (outward)
      5. Épaulement            : de z=-LIP_DEPTH à z=-POCKET_DEPTH sur périmètre corps (outward)
      6. Bouton de préhension  : pavé quad intégré z=0→BUTTON_H_MM (pièce unique, pas de coque séparée)

    Pourquoi un pavé et pas un cylindre ?
      Un cylindre fermé (fond + couvercle) crée DEUX coques distinctes dans le STL.
      Le slicer les imprime comme deux objets → délaminage immédiat.
      Ici le bouton partage la face z=0 du continent → une seule coque continue,
      manifold et sans surface intérieure → pièce unique garantie.
    """
    from scipy.ndimage import distance_transform_edt

    Vo = _make_verts(0.0)
    Vl = _make_verts(-LIP_DEPTH)
    Vi = _make_verts(-POCKET_DEPTH)

    lip_only = lq_lip & ~lq_body

    tris = [
        _top_face(lq_lip,  Vo),          # 1. top complet (= plancher du bouton si add_button)
        _bot_face(lip_only, Vl),         # 2. dessous collerette
        _bot_face(lq_body,  Vi),         # 3. fond corps
        _walls_outward(lq_lip,  Vo, Vl), # 4. parois ext collerette
        _walls_outward(lq_body, Vl, Vi), # 5. épaulement corps
    ]

    # 6) Bouton intégré : pavé quad aligné sur la grille, surface continue avec le continent.
    #    Pas de fond séparé — la face z=0 (tris[0]) constitue déjà le plancher du bouton.
    if add_button and BUTTON_R_MM > 0.0 and BUTTON_H_MM > 0.0:
        dist = distance_transform_edt(lq_body)
        if dist.size and dist.max() > 1.0:
            si, sj = np.unravel_index(np.argmax(dist), dist.shape)
            n_qx = max(1, round(BUTTON_R_MM / (MAP_W / GRID_X)))
            n_qy = max(1, round(BUTTON_R_MM / (MAP_H / GRID_Y)))
            btn = np.zeros((GRID_Y, GRID_X), dtype=bool)
            btn[max(0, si - n_qy):min(GRID_Y, si + n_qy + 1),
                max(0, sj - n_qx):min(GRID_X, sj + n_qx + 1)] = True
            btn &= lq_body   # rester dans les limites du continent
            if btn.any():
                V_btn = _make_verts(BUTTON_H_MM)
                tris += [
                    _top_face(btn, V_btn),           # toit du bouton  (z = BUTTON_H_MM)
                    _walls_outward(btn, V_btn, Vo),  # parois latérales (z = BUTTON_H_MM → 0)
                    # plancher = face sup continent déjà dans tris[0], pas de doublon
                ]

    return np.concatenate(tris, axis=0).astype(np.float32)


# ── Sauvegarde STL ─────────────────────────────────────────────────────────────

def _check_manifold(triangles: np.ndarray) -> int:
    """
    Compte les arêtes dont la multiplicité ≠ 2 (mesh non-manifold).
    Entièrement vectorisé (numpy), ~10× plus rapide que la version Counter/Python.
    Retourne le nombre d'arêtes problématiques (0 = mesh valide).
    """
    q = 1e-6
    v = np.round(triangles.reshape(-1, 3) / q).astype(np.int64).reshape(-1, 3, 3)
    n = len(v)
    # 3 paires d'arêtes (indices locaux) par triangle
    pairs = [(0, 1), (1, 2), (2, 0)]
    all_edges = []
    for i, j in pairs:
        a = v[:, i, :]   # (n, 3)
        b = v[:, j, :]   # (n, 3)
        # Tri lexicographique des 2 sommets : on compare x, puis y, puis z
        gt = (a[:, 0] > b[:, 0]) | \
             ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1])) | \
             ((a[:, 0] == b[:, 0]) & (a[:, 1] == b[:, 1]) & (a[:, 2] > b[:, 2]))
        e0 = np.where(gt[:, None], b, a)   # sommet "min" en premier
        e1 = np.where(gt[:, None], a, b)
        all_edges.append(np.concatenate([e0, e1], axis=1))  # (n, 6)
    flat = np.concatenate(all_edges, axis=0)   # (3n, 6)
    _, counts = np.unique(flat, axis=0, return_counts=True)
    return int((counts != 2).sum())


def save_stl(triangles, path: Path):
    n = len(triangles)
    m = mesh.Mesh(np.zeros(n, dtype=mesh.Mesh.dtype))
    m.vectors[:] = triangles
    m.save(str(path))
    size_mb = path.stat().st_size / 1024 / 1024
    bad = _check_manifold(triangles)
    manifold_str = "OK manifold" if bad == 0 else f"ATTENTION {bad} arêtes non-manifold"
    print(f"  {n:>10,} triangles  |  {size_mb:.1f} MB  |  {manifold_str}  →  {path.name}")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Flat Map 2D STL Generator — Puzzle d'encastrement")
    print(f"  Carte      : {MAP_W} x {MAP_H} mm  (équirectangulaire, max {MAP_MAX_LEN} mm)")
    print(f"  Maillage   : {GRID_X} x {GRID_Y} = {GRID_X*GRID_Y:,} quads")
    print(f"  Épaisseur plaque : {BASE_THICK} mm  |  Profondeur creux : {POCKET_DEPTH} mm")
    print(f"  Jeu        : {INSERT_CLEARANCE} mm  |  Collerette : {LIP_W} x {LIP_DEPTH} mm")
    print(f"  Rebord     : {BORDER_W_MM} mm (largeur) x {BORDER_H_MM} mm (hauteur)")
    print(f"  Bouton     : R {BUTTON_R_MM} mm x H {BUTTON_H_MM} mm")
    print(f"  Coupures géo : {'OUI' if ENABLE_GEOGRAPHIC_CUTS else 'NON'}")
    print(f"  Miroir X/Y : {'OUI' if MIRROR_X else 'NON'} / {'OUI' if MIRROR_Y else 'NON'}")
    print(f"  Base en 2 pièces : {'OUI' if SPLIT_BASE_IN_TWO else 'NON'}")
    print("=" * 62)

    print("\n[1/4] Rasterisation du SVG…")
    land_map, red_mask = rasterize_svg(SVG_FILE, IMG_W, IMG_H)
    land_pct = (land_map < LAND_THRESHOLD).mean() * 100
    print(f"  {IMG_W}×{IMG_H} px  |  {land_pct:.1f}% terres")

    print("\n[2/4] Segmentation des continents…")
    continent_data = label_continents(land_map, red_mask)
    print(f"  {len(continent_data)} masse(s) continentale(s) détectée(s)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cont_dir = OUTPUT_DIR / "continents"
    cont_dir.mkdir(exist_ok=True)

    print("\n[3/4] Plaque de base…")
    if SPLIT_BASE_IN_TWO:
        left_mask, right_mask = _split_footprints()
        left_tris = build_base_plate(continent_data, left_mask)
        right_tris = build_base_plate(continent_data, right_mask)
        save_stl(left_tris, OUTPUT_DIR / "map_base_left.stl")
        save_stl(right_tris, OUTPUT_DIR / "map_base_right.stl")
    else:
        base_tris = build_base_plate(continent_data)
        save_stl(base_tris, OUTPUT_DIR / "map_base.stl")

    print("\n[4/4] Pièces continents…")
    # Pour chaque continent majeur, seul le PLUS GRAND fragment reçoit un bouton.
    from collections import defaultdict
    frag_sizes: dict = defaultdict(list)
    for name, lq_body, lq_exact, lq_lip in continent_data:
        parts = name.rsplit('_', 1)
        base = parts[0] if len(parts) > 1 and parts[1].isdigit() else name
        frag_sizes[base].append((lq_body.sum(), name))
    button_names: set = set()
    for base, frags in frag_sizes.items():
        if base in BUTTON_CONTINENTS:
            largest_name = max(frags, key=lambda x: x[0])[1]
            button_names.add(largest_name)

    for name, lq_body, lq_exact, lq_lip in continent_data:
        tris = build_continent_insert(lq_body, lq_lip, add_button=(name in button_names))
        save_stl(tris, cont_dir / f"{name}.stl")

    print(f"\nDone. Fichiers dans : {OUTPUT_DIR}/")
    if SPLIT_BASE_IN_TWO:
        print("  map_base_left.stl / map_base_right.stl  → 2 moitiés droites du plateau")
    else:
        print("  map_base.stl   → plaque pleine avec creux  (imprimer en premier)")
    print(f"  continents/    → {len(continent_data)} pièces à encastrer")
    print()
    print("  Conseil impression :")
    print("    - Plaque à plat sur le plateau, côté z=0 vers le haut")
    print("    - Continents à plat, collerette vers le bas (z=0 = face visible)")
