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
SVG_FILE   = Path.home() / "Desktop/globe_terrestre/BlankMap_World_simple3.svg"
OUTPUT_STL        = Path.home() / "Desktop/globe_10cm.stl"
OUTPUT_DIR        = Path.home() / "Desktop/continents"   # one STL per continent
MIN_CONTINENT_QUADS = 80   # ignore islands smaller than this many quads
RADIUS     = 50.0   # mm  (15 cm diameter)
LAND_RAISE = 3    # mm  relief for land masses
LAT_STEPS  = 1200    # latitude divisions  (↑ = contours plus fins, fichiers plus lourds)
LON_STEPS  = 2400   # longitude divisions
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
INSERT_CLEAR = 0.1   # mm — jeu radial des inserts continent (pour qu'ils rentrent dans les creux)
# ──────────────────────────────────────────────────────────────────────────────


def rasterize_svg(svg_path: Path, width: int, height: int) -> np.ndarray:
    """Return a (height, width) uint8 grayscale array."""

    # cairosvg (installed via pip)
    try:
        import cairosvg
        data = cairosvg.svg2png(
            url=str(svg_path), output_width=width, output_height=height
        )
        img_rgba = Image.open(io.BytesIO(data)).convert("RGBA")
        # Composite on white background (transparent → white, not black)
        white = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        white.paste(img_rgba, mask=img_rgba.split()[3])
        return np.array(white.convert("L"))
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
            img_rgba = Image.open(tmp).convert("RGBA")
            white = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
            white.paste(img_rgba, mask=img_rgba.split()[3])
            return np.array(white.convert("L"))
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
            img_rgba = Image.open(tmp).convert("RGBA")
            white = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
            white.paste(img_rgba, mask=img_rgba.split()[3])
            return np.array(white.convert("L"))
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
    from scipy.ndimage import gaussian_filter
    img_h, img_w = land_map.shape
    is_land = (land_map < LAND_THRESHOLD).astype(np.float32)
    if smooth and SMOOTH_SIGMA > 0:
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
    flag = _land_flag(land_map, smooth=True)
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
    return np.concatenate([dome, cap, walls, bot]).astype(np.float32)


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
      - Monts Oural       → Europe / Asie (Russia européenne vs Sibérie)
      - Caucase           → relie l'Oural à la mer Noire
    """
    img = land_map.copy()

    # ── Isthme de Panama (~9 °N, 79 °O) ──────────────────────────────────────
    _draw_cut(img, 9.5, -80.0,  8.5, -77.0, width_px=3)

    # ── Canal de Suez (~30 °N, 32 °E) ────────────────────────────────────────
    _draw_cut(img, 32.0, 32.5,  28.5, 32.5, width_px=3)

    # ── Oural : mer de Kara → mer Caspienne ──────────────────────────────────
    _draw_cut(img, 68.0, 60.0,  51.0, 59.0, width_px=4)   # Oural N → Oural S
    _draw_cut(img, 51.0, 59.0,  47.0, 51.5, width_px=4)   # Oural S → Caspienne

    # ── Caucase : Caspienne → mer Noire ──────────────────────────────────────
    _draw_cut(img, 43.0, 48.0,  41.5, 41.0, width_px=4)

    return img


def _mesh_from_lq(lq: np.ndarray, Vo: np.ndarray, Vi: np.ndarray) -> np.ndarray:
    """
    Build a closed shell mesh for the boolean quad mask `lq`.
    Outer face at Vo, inner face at Vi, side walls at every land/ocean edge.
    """
    tris = []

    si, sj = np.where(lq)
    v00o, v10o, v11o, v01o = Vo[si,sj], Vo[si+1,sj], Vo[si+1,sj+1], Vo[si,sj+1]
    v00i, v10i, v11i, v01i = Vi[si,sj], Vi[si+1,sj], Vi[si+1,sj+1], Vi[si,sj+1]

    tris += [np.stack([v00o, v10o, v11o], axis=1),   # outer
             np.stack([v00o, v11o, v01o], axis=1),
             np.stack([v00i, v11i, v10i], axis=1),   # inner (flipped)
             np.stack([v00i, v01i, v11i], axis=1)]

    # Top wall  (ocean at i-1)
    top_oce = np.empty_like(lq); top_oce[0,:] = True; top_oce[1:,:] = ~lq[:-1,:]
    si, sj = np.where(lq & top_oce)
    tris += [np.stack([Vo[si,sj],   Vo[si,sj+1], Vi[si,sj+1]], axis=1),
             np.stack([Vo[si,sj],   Vi[si,sj+1], Vi[si,sj  ]], axis=1)]

    # Bottom wall  (ocean at i+1)
    bot_oce = np.empty_like(lq); bot_oce[-1,:] = True; bot_oce[:-1,:] = ~lq[1:,:]
    si, sj = np.where(lq & bot_oce)
    tris += [np.stack([Vo[si+1,sj+1], Vo[si+1,sj  ], Vi[si+1,sj  ]], axis=1),
             np.stack([Vo[si+1,sj+1], Vi[si+1,sj  ], Vi[si+1,sj+1]], axis=1)]

    # Left wall  (ocean at j-1, seam wraps)
    si, sj = np.where(lq & ~np.roll(lq, 1, axis=1))
    tris += [np.stack([Vo[si+1,sj], Vo[si,  sj], Vi[si,  sj]], axis=1),
             np.stack([Vo[si+1,sj], Vi[si,  sj], Vi[si+1,sj]], axis=1)]

    # Right wall  (ocean at j+1, seam wraps)
    si, sj = np.where(lq & ~np.roll(lq, -1, axis=1))
    tris += [np.stack([Vo[si,  sj+1], Vo[si+1,sj+1], Vi[si+1,sj+1]], axis=1),
             np.stack([Vo[si,  sj+1], Vi[si+1,sj+1], Vi[si,  sj+1]], axis=1)]

    return np.concatenate(tris, axis=0).astype(np.float32)


def build_continent_inserts(land_map: np.ndarray):
    """
    Return a list of (filename_stem, triangles) — one closed mesh per connected
    land region (continent / island).  Requires scipy.
    """
    from scipy.ndimage import label as ndlabel, binary_erosion, binary_closing, binary_opening

    # Éroder le bitmap de la carte de INSERT_CLEAR mm autour du périmètre :
    # la face supérieure des inserts reste à RADIUS (affleure l'océan),
    # seul le contour latéral rétrécit → jeu pour l'encastrement.
    # Érosion du périmètre de INSERT_CLEAR mm pour le jeu d'encastrement
    px_mm = 2 * np.pi * RADIUS / land_map.shape[1]
    n_px  = max(1, round(INSERT_CLEAR / px_mm))
    is_land_eroded = binary_erosion(land_map < LAND_THRESHOLD, iterations=n_px)
    land_map_insert = np.where(is_land_eroded, 0, 255).astype(np.uint8)

    flag = _land_flag(land_map_insert)
    lq = ((flag[:LAT_STEPS, :LON_STEPS] +
           flag[1:,          :LON_STEPS] +
           flag[:LAT_STEPS,  1:        ] +
           flag[1:,          1:        ]) / 4.0) > 0.5

    # Lissage morphologique du masque de quads
    _s = np.ones((3, 3), dtype=bool)
    lq = binary_opening(binary_closing(lq, _s), _s)

    # ── Coupures géographiques ────────────────────────────────────────────────────
    # Les coupures sont directement dessinées dans le SVG source (rectangles blancs
    # ajoutés dans Inkscape aux frontières Panama / Suez / Caucase / Oural).
    # Aucun traitement supplémentaire nécessaire ici.
    # ─────────────────────────────────────────────────────────────────────────────

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
    Vi = sphere_verts(RADIUS - LAND_RAISE)

    # Label connected components (8-connectivity to link diagonally touching quads)
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
            if mask[ci, cj]:
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
        tris = _mesh_from_lq(mask, Vo, Vi)
        results.append((name, tris))
        print(f"    {name:20s}: {mask.sum():>8,} quads")

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
    land_map = rasterize_svg(SVG_FILE, IMG_W, IMG_H)
    land_pct = (land_map < LAND_THRESHOLD).mean() * 100
    print(f"  {IMG_W}x{IMG_H} px  |  {land_pct:.1f}% land detected")

    print("\n[2/4] Building globe hemispheres…")
    north_tris = build_globe_north(land_map)
    south_tris = build_globe_south(land_map)
    print(f"  Nord  : {len(north_tris):,} triangles")
    print(f"  Sud   : {len(south_tris):,} triangles")

    print("\n[3/4] Building continent inserts (composantes connexes)…")
    inserts = build_continent_inserts(land_map)
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
