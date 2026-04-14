#!/usr/bin/env python3
"""
Globe STL Generator
Converts BlankMap_World_simple2.svg to a 3D-printable globe.

Sphere diameter : 150 mm (15 cm)
Land masses     : raised 1.5 mm above ocean surface
Mesh resolution : 300 lat x 600 lon = 360 000 triangles
"""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from stl import mesh

# ── Configuration ─────────────────────────────────────────────────────────────
SVG_FILE   = Path.home() / "/Users/souchaud/Documents/Impression 3D/globe_terrestre/BlankMap_World_simple3.svg"
# OUTPUT_STL        = Path.home() / "Desktop/globe_10cm.stl"
OUTPUT_DIR        = Path.home() / "Desktop/continents"   # one STL per continent
MIN_CONTINENT_QUADS = 40   # ignore islands smaller than this many quads
RADIUS     = 75.0   # mm  (15 cm diameter)
LAND_RAISE = 5    # mm  relief for land masses
LAT_STEPS  = 600    # latitude divisions  (↑ = contours plus fins, fichiers plus lourds)
LON_STEPS  = 1200   # longitude divisions
IMG_W      = 2000   # rasterisation width  (px)
IMG_H      = 1000   # rasterisation height (px)
LAND_THRESHOLD = 230  # grayscale < this → land (#c8c8c8 ≈ 200, background = 255)
SMOOTH_SIGMA   = 4.0  # px — flou gaussien sur les bords (0 = pas de lissage, ↑ = plus doux)
# ── Snap-fit tenon/mortaise à l'équateur ──────────────────────────────────────
SNAP_R       = 8.0   # mm — rayon du cylindre de tenon
SNAP_H       = 8.0   # mm — hauteur/profondeur du tenon (doit être > 2×SNAP_RIDGE_W)
SNAP_CLEAR   = 0.2   # mm — jeu d'assemblage (mortaise = SNAP_R + SNAP_CLEAR)
SNAP_RIDGE   = 0.4   # mm — hauteur du bourrelet de clippage sur le tenon
SNAP_RIDGE_W = 2.0   # mm — demi-largeur du profil du bourrelet
INSERT_CLEAR = -0.1  # mm — négatif = léger serrage (plus précis sur les petits détails)
LIP_W        = 0.6   # mm — largeur de la collerette (débord fin, suit mieux les détails)
LIP_DEPTH    = 1.0   # mm — épaisseur de la collerette (profondeur sous la face visible)
# Continents à forcer en ajustement exact (corps = forme du trou, jupe légère)
# "*" = tous les continents / îles en exact-fit
EXACT_FIT_CONTINENTS = {"*"}
# Activer/désactiver les coupures rouges du SVG (traits .st0 / stroke:red).
# False = les traits rouges sont ignorés (pas de trou/coupure imposé par ces traits).
ENABLE_RED_CUTS = False
# Activer/désactiver les coupures géographiques automatiques (Suez, Gibraltar, Oural, ...)
# False = respecter strictement le SVG (pas de trous/coupures ajoutés par le code).
ENABLE_GEOGRAPHIC_CUTS = False
# ── Trou de support (tube vertical, inclinaison axiale terrestre) ─────────────
SUPPORT_TUBE_RADIUS = 2.6   # mm — rayon du trou (tube Ø 5 mm + 0.2 mm jeu)
SUPPORT_HOLE_DEPTH  = 30.0  # mm — profondeur du trou depuis la surface du globe
AXIAL_TILT_DEG      = 23.5  # °  — obliquité de l'écliptique (inclinaison réelle de la Terre)
# ── Principe de rétention : la collerette (> trou) repose sur la surface globe ─
# Insert impossible à faire tomber ; retrait uniquement par poussée de l'intérieur
# (après ouverture des deux hémisphères via le snap-fit équatorial).
# ──────────────────────────────────────────────────────────────────────────────


def _rgba_to_gray_and_red(img_rgba: Image.Image):
    """Composite RGBA sur blanc, détecte pixels rouges, retourne (gray, red_mask)."""
    white = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    white.paste(img_rgba, mask=img_rgba.split()[3])
    rgb = np.array(white.convert("RGB"))
    red_mask = (rgb[:, :, 0] > 100) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80) & (rgb[:, :, 0] > rgb[:, :, 1] * 2) & (rgb[:, :, 0] > rgb[:, :, 2] * 2)
    rgb[red_mask] = 255
    return np.array(Image.fromarray(rgb).convert("L")), red_mask


def rasterize_svg(svg_path: Path, width: int, height: int):
    """Return (grayscale uint8 array, red_mask bool array)."""

    # cairosvg (installed via pip)
    try:
        import cairosvg
        data = cairosvg.svg2png(
            url=str(svg_path), output_width=width, output_height=height
        )
        return _rgba_to_gray_and_red(Image.open(io.BytesIO(data)).convert("RGBA"))
    except Exception as e:
        print(f"  cairosvg failed ({e}), trying rsvg-convert…")

    # rsvg-convert (brew install librsvg)
    tmp = Path("/tmp/_globe_map.png")
    try:
        r = subprocess.run(
            ["rsvg-convert", "-w", str(width), "-h", str(height),
             "-o", str(tmp), str(svg_path)],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0:
            return _rgba_to_gray_and_red(Image.open(tmp).convert("RGBA"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Inkscape headless
    try:
        r = subprocess.run(
            ["inkscape", "--export-type=png",
             f"--export-filename={tmp}",
             f"--export-width={width}", f"--export-height={height}",
             str(svg_path)],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0:
            return _rgba_to_gray_and_red(Image.open(tmp).convert("RGBA"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise RuntimeError(
        "Could not rasterize the SVG. "
        "Install cairosvg (pip) or librsvg (brew install librsvg)."
    )


def _build_sphere(r: np.ndarray) -> np.ndarray:
    """
    Given a (LAT_STEPS+1, LON_STEPS+1) radius array, return a manifold UV-sphere
    mesh as (N, 3, 3) float32 triangles.
    """
    lats = np.linspace( np.pi / 2, -np.pi / 2, LAT_STEPS + 1)
    lons = np.linspace(-np.pi,      np.pi,      LON_STEPS + 1)

    cos_lat = np.cos(lats)[:, None]
    sin_lat = np.sin(lats)[:, None]
    cos_lon = np.cos(lons)[None, :]
    sin_lon = np.sin(lons)[None, :]

    X = r * cos_lat * cos_lon
    Y = r * cos_lat * sin_lon
    Z = r * sin_lat

    V = np.stack([X, Y, Z], axis=-1)  # (LAT+1, LON+1, 3)

    # Fix non-manifold edges:
    # 1. Close the seam (lon = ±π is the same meridian)
    V[:, -1] = V[:, 0]
    # 2. Collapse each pole row to a single point
    V[0, :]  = V[0,  0]
    V[-1, :] = V[-1, 0]

    ii, jj = np.meshgrid(np.arange(LAT_STEPS), np.arange(LON_STEPS), indexing="ij")
    v00 = V[ii,     jj    ]
    v10 = V[ii + 1, jj    ]
    v11 = V[ii + 1, jj + 1]
    v01 = V[ii,     jj + 1]

    N = LAT_STEPS * LON_STEPS
    tri1 = np.stack([v00, v10, v11], axis=2).reshape(N, 3, 3)
    tri2 = np.stack([v00, v11, v01], axis=2).reshape(N, 3, 3)
    return np.concatenate([tri1, tri2], axis=0).astype(np.float32)


def _land_flag(land_map: np.ndarray, smooth: bool = False) -> np.ndarray:
    """
    Return the (LAT+1, LON+1) land flag grid.
    smooth=True : flou gaussien → valeurs en [0,1], bords arrondis sur la sphère.
    smooth=False : binaire strict 0/1 (utilisé pour les masques d'inserts).
    """
    img_h, img_w = land_map.shape
    is_land = (land_map < LAND_THRESHOLD).astype(np.float32)
    if smooth and SMOOTH_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        is_land = gaussian_filter(is_land, sigma=SMOOTH_SIGMA)
    lats = np.linspace( np.pi / 2, -np.pi / 2, LAT_STEPS + 1)
    lons = np.linspace(-np.pi,      np.pi,      LON_STEPS + 1)
    j_img = ((lons + np.pi) / (2 * np.pi) * img_w).clip(0, img_w - 1).astype(int)
    i_img = ((np.pi / 2 - lats) / np.pi    * img_h).clip(0, img_h - 1).astype(int)
    return is_land[np.ix_(i_img, j_img)]


def build_globe(land_map: np.ndarray) -> np.ndarray:
    """Ocean at RADIUS, continents recessed by LAND_RAISE (bords lissés)."""
    flag = _land_flag(land_map, smooth=True)
    return _build_sphere(RADIUS - LAND_RAISE * flag)


# ── Snap-fit helpers ──────────────────────────────────────────────────────────

def _snap_pts(r: float, z: float) -> np.ndarray:
    """LON_STEPS points on a circle radius r at height z, angles = globe lons."""
    a = np.linspace(-np.pi, np.pi, LON_STEPS + 1)[:-1]
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(LON_STEPS, z)], axis=1)


def _annular_cap(inner: np.ndarray, outer: np.ndarray, normal_pos_z: bool) -> np.ndarray:
    """Annular strip between two co-planar rings of equal point count."""
    n = len(inner)
    j = np.arange(n);  jn = (j + 1) % n
    if normal_pos_z:        # +Z normal (CCW from above)
        t1 = np.stack([inner[j],  outer[j],   outer[jn]], axis=1)
        t2 = np.stack([inner[j],  outer[jn],  inner[jn]], axis=1)
    else:                   # −Z normal
        t1 = np.stack([inner[j],  outer[jn],  outer[j] ], axis=1)
        t2 = np.stack([inner[j],  inner[jn],  outer[jn]], axis=1)
    return np.concatenate([t1, t2], axis=0)


def _frustum_walls(r0: float, z0: float, r1: float, z1: float, outward: bool) -> np.ndarray:
    """Walls of a frustum (cone tronqué) de (r0,z0) à (r1,z1)."""
    rb = _snap_pts(r0, z0);  rt = _snap_pts(r1, z1)
    j = np.arange(LON_STEPS);  jn = (j + 1) % LON_STEPS
    if outward:
        t1 = np.stack([rb[j], rt[j],  rt[jn]], axis=1)
        t2 = np.stack([rb[j], rt[jn], rb[jn]], axis=1)
    else:
        t1 = np.stack([rb[j], rt[jn], rt[j] ], axis=1)
        t2 = np.stack([rb[j], rb[jn], rt[jn]], axis=1)
    return np.concatenate([t1, t2], axis=0)


def _cylinder_walls(r: float, z0: float, z1: float, outward: bool) -> np.ndarray:
    """Side walls of a cylinder between z0 and z1."""
    return _frustum_walls(r, z0, r, z1, outward)


def _disk(r: float, z: float, normal_pos_z: bool) -> np.ndarray:
    """Solid disk of radius r at height z (fan triangulation)."""
    pts = _snap_pts(r, z)
    j = np.arange(LON_STEPS);  jn = (j + 1) % LON_STEPS
    ctr = np.full((LON_STEPS, 3), [0.0, 0.0, z])
    if normal_pos_z:
        return np.stack([ctr, pts[j],  pts[jn]], axis=1)
    else:
        return np.stack([ctr, pts[jn], pts[j] ], axis=1)


def _globe_verts(land_map: np.ndarray) -> np.ndarray:
    """(LAT+1, LON+1, 3) vertex array for the globe (seam closed, poles NOT collapsed)."""
    flag = _land_flag(land_map, smooth=False)  # parois nettes → frottement avec les inserts
    r = RADIUS - LAND_RAISE * flag
    lats = np.linspace( np.pi / 2, -np.pi / 2, LAT_STEPS + 1)
    lons = np.linspace(-np.pi,      np.pi,      LON_STEPS + 1)
    X = r * np.cos(lats)[:, None] * np.cos(lons)[None, :]
    Y = r * np.cos(lats)[:, None] * np.sin(lons)[None, :]
    Z = r * np.sin(lats)[:, None]
    V = np.stack([X, Y, Z], axis=-1)
    V[:, -1] = V[:, 0]   # close seam
    return V


def _dome_tris(V: np.ndarray, i_start: int, i_end: int) -> np.ndarray:
    """Triangles for globe quads from row i_start to i_end-1."""
    ii, jj = np.meshgrid(np.arange(i_start, i_end), np.arange(LON_STEPS), indexing="ij")
    v00, v10, v11, v01 = V[ii, jj], V[ii+1, jj], V[ii+1, jj+1], V[ii, jj+1]
    N = (i_end - i_start) * LON_STEPS
    t1 = np.stack([v00, v10, v11], axis=2).reshape(N, 3, 3)
    t2 = np.stack([v00, v11, v01], axis=2).reshape(N, 3, 3)
    return np.concatenate([t1, t2], axis=0)


def build_globe_north(land_map: np.ndarray) -> np.ndarray:
    """
    Hémisphère Nord (Z ≥ 0) avec tenon mâle cylindrique à l'équateur.
    Imprimer à plat (face plate sur le plateau).
    """
    V = _globe_verts(land_map)
    V[0, :] = V[0, 0]   # pôle Nord → point unique

    eq = LAT_STEPS // 2
    dome = _dome_tris(V, 0, eq)
    cap  = _annular_cap(_snap_pts(SNAP_R, 0.0), V[eq, :LON_STEPS], normal_pos_z=False)

    # Tenon avec bourrelet de clippage :
    #   fût lisse → rampe montante → sommet du bourrelet → rampe descendante → fond
    r_rid = SNAP_R + SNAP_RIDGE          # rayon au sommet du bourrelet
    z_a   = -(SNAP_H - 2 * SNAP_RIDGE_W) # début de la rampe montante
    z_b   = -(SNAP_H - SNAP_RIDGE_W)     # sommet du bourrelet
    z_bot = -SNAP_H
    walls = np.concatenate([
        _cylinder_walls(SNAP_R, z0=0.0, z1=z_a,  outward=True),   # fût
        _frustum_walls( SNAP_R, z_a,    r_rid, z_b,   outward=True),  # rampe ↑
        _frustum_walls( r_rid,  z_b,    SNAP_R, z_bot, outward=True),  # rampe ↓
    ])
    bot = _disk(SNAP_R, z=z_bot, normal_pos_z=False)
    return np.concatenate([dome, cap, walls, bot]).astype(np.float32)


def build_globe_south(land_map: np.ndarray) -> np.ndarray:
    """
    Hémisphère Sud (Z ≤ 0) avec mortaise (socket) cylindrique à l'équateur.
    Imprimer à plat (face plate sur le plateau).
    """
    V = _globe_verts(land_map)
    V[-1, :] = V[-1, 0]  # pôle Sud → point unique

    eq     = LAT_STEPS // 2
    r_sock = SNAP_R + SNAP_CLEAR
    dome   = _dome_tris(V, eq, LAT_STEPS)
    cap    = _annular_cap(_snap_pts(r_sock, 0.0), V[eq, :LON_STEPS], normal_pos_z=True)

    # Mortaise avec gorge : creusement au même z que le bourrelet du tenon
    r_grv = SNAP_R + SNAP_RIDGE + SNAP_CLEAR  # rayon de la gorge (reçoit le bourrelet)
    z_a   = -(SNAP_H - 2 * SNAP_RIDGE_W)
    z_b   = -(SNAP_H - SNAP_RIDGE_W)
    z_bot = -SNAP_H
    walls = np.concatenate([
        _cylinder_walls(r_sock, z0=0.0, z1=z_a,  outward=False),   # paroi lisse
        _frustum_walls( r_sock, z_a,    r_grv, z_b,    outward=False),  # gorge s'ouvre
        _frustum_walls( r_grv,  z_b,    r_sock, z_bot,  outward=False),  # gorge se ferme
    ])
    bot = _disk(r_sock, z=z_bot, normal_pos_z=True)
    south = np.concatenate([dome, cap, walls, bot]).astype(np.float32)
    return _add_support_hole(south)


def _add_support_hole(triangles: np.ndarray) -> np.ndarray:
    """
    Perce un trou cylindrique borgne dans l'hémisphère Sud pour accueillir un
    tube de support vertical de Ø 5 mm.

    Géométrie :
      - Le globe est incliné de AXIAL_TILT_DEG (23,5°) sur son support.
      - Le trou est aligné avec un tube vertical → dans le repère du globe,
        le trou pointe à 23,5° de l'axe polaire Sud.
      - Point d'entrée = point le plus bas du globe sur son support.
      - Profondeur : SUPPORT_HOLE_DEPTH mm.

    Requiert : pip install trimesh manifold3d
    """
    try:
        import manifold3d as md
        import trimesh
    except ImportError:
        print("  ⚠ manifold3d/trimesh non installé — trou de support ignoré.")
        print("    → pip install trimesh manifold3d")
        return triangles

    tilt = np.radians(AXIAL_TILT_DEG)

    # Direction "monde vertical" dans le repère du globe incliné à 23,5°.
    # Le cylindre (axe Z par défaut) est pivoté de tilt° autour de Y.
    d = np.array([np.sin(tilt), 0.0, np.cos(tilt)])   # vecteur sortant du trou

    # Point d'entrée du trou sur la surface de la sphère
    p_entry = -RADIUS * d

    # Le cylindre dépasse de 5 mm hors de la sphère pour un perçage propre
    cyl_len  = SUPPORT_HOLE_DEPTH + 5.0
    p_start  = p_entry - 5.0 * d                      # 5 mm hors sphère
    cyl_ctr  = p_start + (cyl_len / 2.0) * d          # centre du cylindre

    # ── Créer le cylindre via manifold3d ──────────────────────────────────────
    # cylinder(height, radius_low) centré à l'origine, axe Z
    cyl = md.Manifold.cylinder(
        height=cyl_len,
        radius_low=SUPPORT_TUBE_RADIUS,
        circular_segments=64,
        center=True,
    )
    # Pivoter de tilt° autour de Y → aligne l'axe Z du cylindre sur d
    cyl = cyl.rotate([0.0, float(np.degrees(tilt)), 0.0])
    # Translater vers le bon emplacement
    cyl = cyl.translate(cyl_ctr.tolist())

    # ── Convertir les triangles numpy en Manifold ─────────────────────────────
    # Merger les sommets dupliqués (triangle soup → mesh indexé)
    verts_raw = triangles.reshape(-1, 3).astype(np.float32)
    faces_raw = np.arange(len(verts_raw), dtype=np.uint32).reshape(-1, 3)
    tm_tmp = trimesh.Trimesh(vertices=verts_raw, faces=faces_raw, process=True)
    tm_tmp.merge_vertices(digits_vertex=4)
    tm_tmp.remove_unreferenced_vertices()
    verts = tm_tmp.vertices.astype(np.float32)
    faces = tm_tmp.faces.astype(np.uint32)
    globe_m = md.Manifold(md.Mesh(vert_properties=verts, tri_verts=faces))

    if globe_m.status() != md.Error.NoError:
        print(f"  ⚠ Mesh invalide ({globe_m.status()}) — trou ignoré.")
        return triangles

    # ── Soustraction booléenne ────────────────────────────────────────────────
    result = globe_m - cyl

    if result.is_empty() or result.num_tri() == 0:
        print("  ⚠ Résultat vide — trou de support ignoré.")
        return triangles

    # ── Convertir le résultat en triangles numpy ──────────────────────────────
    rm = result.to_mesh()
    r_verts = np.array(rm.vert_properties, dtype=np.float32)
    r_faces = np.array(rm.tri_verts,       dtype=np.int32)
    out = r_verts[r_faces]   # (N, 3, 3) — tableau de triangles

    print(f"  Trou de support : Ø {SUPPORT_TUBE_RADIUS * 2:.1f} mm  |  "
          f"profondeur {SUPPORT_HOLE_DEPTH:.0f} mm  |  inclinaison {AXIAL_TILT_DEG}°")
    return out.astype(np.float32)


def _draw_cut(img: np.ndarray, lat0: float, lon0: float,
              lat1: float, lon1: float, width_px: int = 4) -> None:
    """Trace une ligne 'océan' (255) entre deux coordonnées géographiques."""
    h, w = img.shape
    x0 = int((lon0 + 180) / 360 * w);  y0 = int((90 - lat0) / 180 * h)
    x1 = int((lon1 + 180) / 360 * w);  y1 = int((90 - lat1) / 180 * h)
    n  = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.round(np.linspace(x0, x1, n)).astype(int)
    ys = np.round(np.linspace(y0, y1, n)).astype(int)
    for dy in range(-width_px, width_px + 1):
        for dx in range(-width_px, width_px + 1):
            img[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] = 255


def _apply_geographic_cuts(land_map: np.ndarray) -> np.ndarray:
    """
    Trace des coupures 'océan' aux frontières classiques afin de séparer
    les masses continentales en pièces de puzzle distinctes.

    Coupures :
      - Isthme de Panama  → Amérique du Nord / Amérique du Sud
      - Canal de Suez     → Afrique / Asie
      - Détroit de Gibraltar → Afrique / Europe (anti-pontage)
      - Monts Oural       → Europe / Asie (Russia européenne vs Sibérie)
      - Caucase           → relie l'Oural à la mer Noire
      - Bosphore / Dardanelles → Europe / Asie (anti-pontage)
    """
    img = land_map.copy()

    # ── Isthme de Panama (~9 °N, 79 °O) ──────────────────────────────────────
    _draw_cut(img, 9.5, -80.0,  8.5, -77.0, width_px=3)

    # ── Canal de Suez (~30 °N, 32 °E) ────────────────────────────────────────
    _draw_cut(img, 32.0, 32.5,  28.5, 32.5, width_px=3)

    # ── Détroit de Gibraltar (~36 °N, 5.5 °O) ────────────────────────────────
    _draw_cut(img, 36.2, -6.2, 35.5, -4.6, width_px=3)

    # ── Oural : mer de Kara → mer Caspienne ──────────────────────────────────
    _draw_cut(img, 68.0, 60.0,  51.0, 59.0, width_px=4)   # Oural N → Oural S
    _draw_cut(img, 51.0, 59.0,  47.0, 51.5, width_px=4)   # Oural S → Caspienne

    # ── Caucase : Caspienne → mer Noire ──────────────────────────────────────
    _draw_cut(img, 43.0, 48.0,  41.5, 41.0, width_px=4)

    # ── Bosphore / Dardanelles (~41 °N, 29 °E) ───────────────────────────────
    _draw_cut(img, 41.4, 29.2, 40.2, 26.0, width_px=3)

    return img


def _mesh_from_lq(lq: np.ndarray, Vo: np.ndarray, Vi: np.ndarray,
                  lq_lip: np.ndarray = None,
                  Vl: np.ndarray = None) -> np.ndarray:
    """
    Build a closed shell mesh for continent insert with optional collerette.

    lq     : quad mask for the insert body (walls + inner face) — fits inside hole.
    Vo     : outer vertex array at RADIUS (visible face).
    Vi     : inner vertex array at RADIUS − LAND_RAISE (floor of insert).
    lq_lip : quad mask for the outer visible face — wider than lq (collerette).
             If None → no collerette (same as lq).
    Vl     : vertex array at RADIUS − LIP_DEPTH (underside of collerette).
             If None → defaults to Vo (zero-depth collerette, backward-compat).

    Rétention : la collerette (lq_lip > trou) repose sur la surface océan du globe.
    L'insert ne peut pas tomber ; retrait par poussée de l'intérieur uniquement.
    """
    if lq_lip is None:
        lq_lip = lq
    if Vl is None:
        Vl = Vo

    tris = []

    # ── 1. Face extérieure (collerette + corps) ────────────────────────────────
    si, sj = np.where(lq_lip)
    tris += [np.stack([Vo[si, sj],   Vo[si+1, sj],   Vo[si+1, sj+1]], axis=1),
             np.stack([Vo[si, sj],   Vo[si+1, sj+1], Vo[si,   sj+1]], axis=1)]

    # ── 2. Face intérieure du corps ────────────────────────────────────────────
    si, sj = np.where(lq)
    tris += [np.stack([Vi[si, sj],   Vi[si+1, sj+1], Vi[si+1, sj]  ], axis=1),
             np.stack([Vi[si, sj],   Vi[si,   sj+1], Vi[si+1, sj+1]], axis=1)]

    # ── 3. Dessous de la collerette (anneau lq_lip ∖ lq) ──────────────────────
    lip_only = lq_lip & ~lq
    if lip_only.any():
        si, sj = np.where(lip_only)
        tris += [np.stack([Vl[si, sj],   Vl[si+1, sj+1], Vl[si+1, sj]  ], axis=1),
                 np.stack([Vl[si, sj],   Vl[si,   sj+1], Vl[si+1, sj+1]], axis=1)]

    # ── Helper : parois latérales entre deux nappes de vertices ───────────────
    def _walls(mask, Ra, Rb):
        ws = []
        # Nord (océan en i−1)
        oce = np.empty_like(mask); oce[0, :] = True; oce[1:, :] = ~mask[:-1, :]
        si, sj = np.where(mask & oce)
        ws += [np.stack([Ra[si, sj],   Ra[si, sj+1], Rb[si, sj+1]], axis=1),
               np.stack([Ra[si, sj],   Rb[si, sj+1], Rb[si, sj  ]], axis=1)]
        # Sud (océan en i+1)
        oce = np.empty_like(mask); oce[-1, :] = True; oce[:-1, :] = ~mask[1:, :]
        si, sj = np.where(mask & oce)
        ws += [np.stack([Ra[si+1, sj+1], Ra[si+1, sj  ], Rb[si+1, sj  ]], axis=1),
               np.stack([Ra[si+1, sj+1], Rb[si+1, sj  ], Rb[si+1, sj+1]], axis=1)]
        # Ouest (seam wraps)
        si, sj = np.where(mask & ~np.roll(mask, 1, axis=1))
        ws += [np.stack([Ra[si+1, sj], Ra[si,   sj], Rb[si,   sj]], axis=1),
               np.stack([Ra[si+1, sj], Rb[si,   sj], Rb[si+1, sj]], axis=1)]
        # Est (seam wraps)
        si, sj = np.where(mask & ~np.roll(mask, -1, axis=1))
        ws += [np.stack([Ra[si,   sj+1], Ra[si+1, sj+1], Rb[si+1, sj+1]], axis=1),
               np.stack([Ra[si,   sj+1], Rb[si+1, sj+1], Rb[si,   sj+1]], axis=1)]
        return ws

    # ── 4. Paroi extérieure de la collerette : Vo → Vl (bord du débord) ───────
    tris += _walls(lq_lip, Vo, Vl)

    # ── 5. Épaulement : périmètre du corps Vl → Vi (liaison collerette→corps) ─
    tris += _walls(lq, Vl, Vi)

    return np.concatenate(tris, axis=0).astype(np.float32)


def build_continent_inserts(land_map: np.ndarray, red_mask: np.ndarray = None):
    """
    Return a list of (filename_stem, triangles) — one closed mesh per connected
    land region (continent / island).  Requires scipy.
    """
    from scipy.ndimage import (
        label as ndlabel,
        binary_erosion,
        binary_dilation,
        binary_opening,
        binary_closing,
    )

    # INSERT_CLEAR > 0 → érosion (jeu, insert plus petit que le creux)
    # INSERT_CLEAR = 0 → ajustement exact
    # INSERT_CLEAR < 0 → dilatation (press-fit, insert plus large = tient sans colle)
    px_mm  = 2 * np.pi * RADIUS / land_map.shape[1]
    n_px   = max(1, round(abs(INSERT_CLEAR) / px_mm))
    land_b = land_map < LAND_THRESHOLD
    if INSERT_CLEAR > 0:
        land_map_insert = np.where(binary_erosion(land_b,  iterations=n_px), 0, 255).astype(np.uint8)
    elif INSERT_CLEAR < 0:
        land_map_insert = np.where(binary_dilation(land_b, iterations=n_px), 0, 255).astype(np.uint8)
    else:
        land_map_insert = np.where(land_b, 0, 255).astype(np.uint8)

    # Masque "fit exact" = même base que le trou du globe (sans INSERT_CLEAR).
    land_map_fit = np.where(land_b, 0, 255).astype(np.uint8)

    # Coupures géographiques automatiques (optionnel).
    if ENABLE_GEOGRAPHIC_CUTS:
        _before_geo_insert = land_map_insert < LAND_THRESHOLD
        _before_geo_fit    = land_map_fit < LAND_THRESHOLD
        land_map_insert = _apply_geographic_cuts(land_map_insert)
        land_map_fit    = _apply_geographic_cuts(land_map_fit)
        geo_cut_px = (_before_geo_insert & ~(land_map_insert < LAND_THRESHOLD)) | (
            _before_geo_fit & ~(land_map_fit < LAND_THRESHOLD)
        )
    else:
        geo_cut_px = np.zeros_like(land_map, dtype=bool)

    # 1) Pixels rouges SVG → océan forcé (avant _land_flag), optionnel
    if ENABLE_RED_CUTS and red_mask is not None:
        land_map_insert[red_mask] = 255
        land_map_fit[red_mask] = 255

    flag = _land_flag(land_map_insert)
    lq = ((flag[:LAT_STEPS, :LON_STEPS] +
           flag[1:,          :LON_STEPS] +
           flag[:LAT_STEPS,  1:        ] +
           flag[1:,          1:        ]) / 4.0) > 0.5
    flag_fit = _land_flag(land_map_fit)
    lq_fit = ((flag_fit[:LAT_STEPS, :LON_STEPS] +
               flag_fit[1:,          :LON_STEPS] +
               flag_fit[:LAT_STEPS,  1:        ] +
               flag_fit[1:,          1:        ]) / 4.0) > 0.5

    # Nettoyage morphologique du masque de quads
    # Ouverture uniquement : retire le bruit sans "recoller" des continents proches.
    _s = np.ones((3, 3), dtype=bool)
    lq = binary_opening(lq, _s)

    # 2) Ré-appliquer les coupures (rouges + géographiques) dans lq
    img_h, img_w = land_map.shape
    _lats_q = np.linspace(np.pi/2, -np.pi/2, LAT_STEPS + 1)
    _lons_q = np.linspace(-np.pi,   np.pi,   LON_STEPS + 1)
    _i_img  = ((np.pi/2 - _lats_q) / np.pi * img_h).clip(0, img_h-1).astype(int)
    _j_img  = ((_lons_q + np.pi) / (2*np.pi) * img_w).clip(0, img_w-1).astype(int)

    if ENABLE_RED_CUTS and red_mask is not None:
        red_q = red_mask[np.ix_(_i_img, _j_img)]
        red_any = (red_q[:LAT_STEPS, :LON_STEPS] | red_q[1:, :LON_STEPS] |
                   red_q[:LAT_STEPS,  1:]         | red_q[1:,  1:])
    else:
        red_any = np.zeros((LAT_STEPS, LON_STEPS), dtype=bool)

    geo_q = geo_cut_px[np.ix_(_i_img, _j_img)]
    geo_any = (geo_q[:LAT_STEPS, :LON_STEPS] | geo_q[1:, :LON_STEPS] |
               geo_q[:LAT_STEPS,  1:]         | geo_q[1:,  1:])

    cut_any = red_any | geo_any
    if cut_any.any():
        lq[cut_any] = False
        lq_fit[cut_any] = False
    if red_any.any():
        print(f"  {red_any.sum():,} quads forcés en océan (coupures rouges SVG)")
    if geo_any.any():
        print(f"  {geo_any.sum():,} quads forcés en océan (coupures géographiques)")

    lats = np.linspace( np.pi/2, -np.pi/2, LAT_STEPS+1)
    lons = np.linspace(-np.pi,    np.pi,   LON_STEPS+1)
    cos_lat = np.cos(lats)[:, None]
    sin_lat = np.sin(lats)[:, None]
    cos_lon = np.cos(lons)[None, :]
    sin_lon = np.sin(lons)[None, :]

    def sphere_verts(r):
        """r peut être un scalaire ou un tableau (LAT+1, LON+1)."""
        V = np.stack([r * cos_lat * cos_lon,
                      r * cos_lat * sin_lon,
                      r * sin_lat * np.ones_like(cos_lon)], axis=-1)
        V[:, -1] = V[:, 0]
        return V

    # Vo = RADIUS partout : la face visible de l'insert est toujours flush avec l'océan.
    # Vi = RADIUS - LAND_RAISE : le fond de l'insert repose sur le plancher du creux.
    # Épaisseur uniforme = LAND_RAISE (3 mm).
    Vo = sphere_verts(RADIUS)
    Vl = sphere_verts(RADIUS - LIP_DEPTH)          # dessous de la collerette
    Vi = sphere_verts(RADIUS - LAND_RAISE)

    # Largeur de la collerette en quads (au moins 1)
    from scipy.ndimage import binary_dilation as _bdil
    _quad_mm = 2 * np.pi * RADIUS / LON_STEPS      # mm par quad à l'équateur
    n_lip    = max(1, round(LIP_W / _quad_mm))

    force_all_exact_fit = "*" in EXACT_FIT_CONTINENTS

    # Segmentation des continents/pièces sur le masque "insert" (lq) uniquement.
    # L'exact-fit s'applique ensuite sur la forme du corps (mask_body = mask & lq_fit)
    # sans fusionner artificiellement des masses continentales.
    labeled, n_comp = ndlabel(lq, structure=np.ones((3, 3)))

    # Seed points pour nommer automatiquement les grandes masses continentales
    CONTINENT_SEEDS = [
        # (lat °N, lon °E, name)
        ( 40.0, -100.0, "north_america"),
        (-15.0,  -55.0, "south_america"),
        (  5.0,   20.0, "africa"),
        ( 50.0,   15.0, "europe"),
        ( 50.0,  100.0, "asia"),
        (-25.0,  135.0, "australia"),
        (-80.0,    0.0, "antarctica"),
        ( 72.0,  -42.0, "greenland"),
    ]

    def _name_component(mask):
        for lat, lon, name in CONTINENT_SEEDS:
            ci = int(np.clip((90 - lat)  / 180 * LAT_STEPS, 0, LAT_STEPS - 2))
            cj = int(np.clip((lon + 180) / 360 * LON_STEPS, 0, LON_STEPS - 2))
            r  = 8  # cherche dans un voisinage de 8 quads (~2mm) autour du seed
            if mask[max(0, ci-r):ci+r+1, max(0, cj-r):cj+r+1].any():
                return name
        return None

    used_names: dict = {}
    results = []
    for cid in range(1, n_comp + 1):
        mask = labeled == cid
        if mask.sum() < MIN_CONTINENT_QUADS:
            continue
        name = _name_component(mask)
        if name is None:
            name = f"island_{cid:03d}"
        elif name in used_names:
            used_names[name] += 1
            name = f"{name}_{used_names[name]}"
        else:
            used_names[name] = 1
        is_exact_fit = force_all_exact_fit or any(
            name == n or name.startswith(f"{n}_") for n in EXACT_FIT_CONTINENTS
        )

        # Cas exact-fit : le corps suit exactement la forme du trou du globe.
        if is_exact_fit:
            mask_body = mask & lq_fit
            if not mask_body.any():  # fallback robuste
                mask_body = mask
        else:
            mask_body = mask

        # Jupe : dilation du contour + lissage léger pour éviter les dents/pics.
        mask_lip = _bdil(mask_body, iterations=n_lip)
        mask_lip = binary_closing(mask_lip, structure=np.ones((3, 3)), iterations=1)

        # Empêche la jupe de recouvrir un autre continent ou une ligne de coupure.
        occupied = lq_fit if is_exact_fit else lq
        mask_lip[occupied & ~mask_body] = False
        mask_lip[cut_any] = False
        mask_lip |= mask_body

        # Biseau de profondeur : 0 (bord ext. de la jupe) → LIP_DEPTH (bord du corps)
        from scipy.ndimage import distance_transform_edt as _edt
        dist_lip = _edt(mask_lip)
        taper_q  = np.clip((dist_lip - 1) / max(n_lip - 1, 1), 0.0, 1.0)
        padded   = np.pad(taper_q, 1, mode='edge')
        taper_v  = (padded[:-1,:-1] + padded[:-1,1:] + padded[1:,:-1] + padded[1:,1:]) / 4
        Vl_sm    = sphere_verts(RADIUS - LIP_DEPTH * taper_v)

        tris = _mesh_from_lq(mask_body, Vo, Vi, lq_lip=mask_lip, Vl=Vl_sm)
        results.append((name, tris))
        fit_tag = " exact-fit" if is_exact_fit else ""
        print(f"    {name:20s}: {mask_body.sum():>8,} quads  (collerette {LIP_W}mm){fit_tag}")

    return results


def save_stl(triangles: np.ndarray, path: Path) -> None:
    n = len(triangles)
    globe = mesh.Mesh(np.zeros(n, dtype=mesh.Mesh.dtype))
    globe.vectors[:] = triangles
    globe.save(str(path))
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  {n:,} triangles  |  {size_mb:.1f} MB  ->  {path}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Globe STL Generator")
    print(f"  Diameter : {RADIUS * 2:.0f} mm")
    print(f"  Relief   : {LAND_RAISE} mm")
    print(f"  Mesh     : {LAT_STEPS} x {LON_STEPS} = {LAT_STEPS*LON_STEPS*2:,} triangles")
    print("=" * 55)

    print("\n[1/3] Rasterising SVG…")
    land_map, red_mask = rasterize_svg(SVG_FILE, IMG_W, IMG_H)
    land_pct = (land_map < LAND_THRESHOLD).mean() * 100
    cut_pct  = red_mask.mean() * 100
    print(f"  {IMG_W}x{IMG_H} px  |  {land_pct:.1f}% land  |  {cut_pct:.3f}% coupures rouges")

    print("\n[2/4] Building globe hemispheres…")
    north_tris = build_globe_north(land_map)
    south_tris = build_globe_south(land_map)
    print(f"  Nord  : {len(north_tris):,} triangles")
    print(f"  Sud   : {len(south_tris):,} triangles")

    print("\n[3/4] Building continent inserts (composantes connexes)…")
    inserts = build_continent_inserts(land_map, red_mask)
    print(f"  {len(inserts)} continent(s) / île(s) détecté(s)")

    print("\n[4/4] Writing STL files…")
    out = Path.home() / "Desktop"
    save_stl(north_tris, out / "globe_nord.stl")
    save_stl(south_tris, out / "globe_sud.stl")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tris in inserts:
        save_stl(tris, OUTPUT_DIR / f"{name}.stl")

    print(f"\nDone.")
    print(f"  globe_nord.stl  → hémisphère Nord  (tenon mâle)")
    print(f"  globe_sud.stl   → hémisphère Sud   (mortaise femelle)")
    print(f"  {OUTPUT_DIR}/ → {len(inserts)} pièces continent")
