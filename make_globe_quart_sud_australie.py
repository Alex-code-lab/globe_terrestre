#!/usr/bin/env python3
"""
Quart de globe Sud + insert Australie.

Version alignée sur make_globe.py :
- mêmes paramètres globaux (rayon, relief, seuils, lissage, jupe),
- même pipeline de génération des inserts (build_continent_inserts),
- donc même forme d'Australie pour l'emboîtement.
"""

from pathlib import Path

import numpy as np
from stl import mesh as stl_mesh

import make_globe as base


OUTPUT_BASE_STL = Path.home() / "Desktop/globe_quart_sud_base.stl"
OUTPUT_INSERT_STL = Path.home() / "Desktop/globe_quart_sud_insert_australie.stl"

# Quart visé : hémisphère Sud + secteur 90E..180E (contient l'Australie).
LAT_MAX_DEG = 0.0
LON_MIN_DEG = 90.0
LON_MAX_DEG = 180.0

# Géométrie: même réglage que make_globe.py (pour compatibilité d'emboîtement)
RADIUS = base.RADIUS
LAND_RAISE = base.LAND_RAISE
CORE_RADIUS = 1.0  # mm (fermeture interne du socle quart)


def _sphere_vertices(r: np.ndarray | float) -> np.ndarray:
    lats = np.linspace(np.pi / 2, -np.pi / 2, base.LAT_STEPS + 1)
    lons = np.linspace(-np.pi, np.pi, base.LON_STEPS + 1)
    cos_lat = np.cos(lats)[:, None]
    sin_lat = np.sin(lats)[:, None]
    cos_lon = np.cos(lons)[None, :]
    sin_lon = np.sin(lons)[None, :]

    V = np.stack(
        [
            r * cos_lat * cos_lon,
            r * cos_lat * sin_lon,
            r * sin_lat * np.ones_like(cos_lon),
        ],
        axis=-1,
    ).astype(np.float32)
    V[:, -1] = V[:, 0]
    return V


def _quarter_mask() -> np.ndarray:
    lat_edges = np.linspace(90.0, -90.0, base.LAT_STEPS + 1)
    lon_edges = np.linspace(-180.0, 180.0, base.LON_STEPS + 1)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    in_south = lat_c <= LAT_MAX_DEG
    in_lon = (lon_c >= LON_MIN_DEG) & (lon_c < LON_MAX_DEG)
    return in_south[:, None] & in_lon[None, :]


def _pick_australia_insert(inserts):
    cands = [(name, tris) for name, tris in inserts
             if name == "australia" or name.startswith("australia_")]
    if not cands:
        raise RuntimeError("Insert 'australia' introuvable dans build_continent_inserts().")
    # Si plusieurs, prend la composante la plus grande.
    return max(cands, key=lambda x: len(x[1]))


def _load_australia_stl_from_output_dir() -> np.ndarray:
    path = base.OUTPUT_DIR / "australia.stl"
    if not path.exists():
        raise RuntimeError(
            "scipy absent et fichier australia.stl introuvable dans OUTPUT_DIR. "
            "Génère d'abord les continents avec make_globe.py (ou installe scipy)."
        )
    m = stl_mesh.Mesh.from_file(str(path))
    return m.vectors.astype(np.float32)


if __name__ == "__main__":
    print("=" * 64)
    print("  Quart Sud + Insert Australie (même pipeline que make_globe.py)")
    print(f"  Rayon={RADIUS} mm  |  relief={LAND_RAISE} mm")
    print("=" * 64)

    print("\n[1/4] Rasterisation carte…")
    land_map, red_mask = base.rasterize_svg(base.SVG_FILE, base.IMG_W, base.IMG_H)

    print("[2/4] Socle quart (relief identique à make_globe.py)…")
    quarter_q = _quarter_mask()
    flag = base._land_flag(land_map, smooth=False)
    Vo_base = _sphere_vertices(RADIUS - LAND_RAISE * flag)
    Vi_base = _sphere_vertices(CORE_RADIUS)
    base_tris = base._mesh_from_lq(quarter_q, Vo_base, Vi_base)
    print(f"  quads quart : {quarter_q.sum():,}")

    print("[3/4] Insert Australie (exactement celui de make_globe.py)…")
    try:
        inserts = base.build_continent_inserts(land_map, red_mask)
        aus_name, ins_tris = _pick_australia_insert(inserts)
        print(f"  insert sélectionné : {aus_name}")
    except ModuleNotFoundError as e:
        if e.name != "scipy":
            raise
        ins_tris = _load_australia_stl_from_output_dir()
        print("  scipy non installé -> utilisation de continents/australia.stl")

    print("[4/4] Export STL…")
    base.save_stl(base_tris, OUTPUT_BASE_STL)
    base.save_stl(ins_tris, OUTPUT_INSERT_STL)

    print("\nFichiers générés :")
    print(f"  - {OUTPUT_BASE_STL}")
    print(f"  - {OUTPUT_INSERT_STL}")
