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
SVG_FILE   = Path.home() / "Desktop/BlankMap_World_simple2.svg"
OUTPUT_STL        = Path.home() / "Desktop/globe_15cm.stl"
OUTPUT_DIR        = Path.home() / "Desktop/continents"   # one STL per continent
MIN_CONTINENT_QUADS = 50   # ignore islands smaller than this many quads
RADIUS     = 50.0   # mm  (15 cm diameter)
LAND_RAISE = 3    # mm  relief for land masses
LAT_STEPS  = 300    # latitude divisions
LON_STEPS  = 600    # longitude divisions
IMG_W      = 2000   # rasterisation width  (px)
IMG_H      = 1000   # rasterisation height (px)
LAND_THRESHOLD = 230  # grayscale < this → land (#c8c8c8 ≈ 200, background = 255)
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


def _land_flag(land_map: np.ndarray) -> np.ndarray:
    """Return the (LAT+1, LON+1) land flag grid from a grayscale land map."""
    img_h, img_w = land_map.shape
    is_land = (land_map < LAND_THRESHOLD).astype(np.float32)
    lats = np.linspace( np.pi / 2, -np.pi / 2, LAT_STEPS + 1)
    lons = np.linspace(-np.pi,      np.pi,      LON_STEPS + 1)
    j_img = ((lons + np.pi) / (2 * np.pi) * img_w).clip(0, img_w - 1).astype(int)
    i_img = ((np.pi / 2 - lats) / np.pi    * img_h).clip(0, img_h - 1).astype(int)
    return is_land[np.ix_(i_img, j_img)]


def build_globe(land_map: np.ndarray) -> np.ndarray:
    """Ocean at RADIUS, continents recessed by LAND_RAISE."""
    flag = _land_flag(land_map)
    return _build_sphere(RADIUS - LAND_RAISE * flag)


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
    from scipy.ndimage import label as ndlabel

    flag = _land_flag(land_map)
    lq = ((flag[:LAT_STEPS, :LON_STEPS] +
           flag[1:,          :LON_STEPS] +
           flag[:LAT_STEPS,  1:        ] +
           flag[1:,          1:        ]) / 4.0) > 0.5

    lats = np.linspace( np.pi/2, -np.pi/2, LAT_STEPS+1)
    lons = np.linspace(-np.pi,    np.pi,   LON_STEPS+1)
    cos_lat = np.cos(lats)[:, None]
    sin_lat = np.sin(lats)[:, None]
    cos_lon = np.cos(lons)[None, :]
    sin_lon = np.sin(lons)[None, :]

    def sphere_verts(radius):
        V = np.stack([radius * cos_lat * cos_lon,
                      radius * cos_lat * sin_lon,
                      radius * sin_lat * np.ones_like(cos_lon)], axis=-1)
        V[:, -1] = V[:, 0]
        return V

    Vo = sphere_verts(RADIUS)
    Vi = sphere_verts(RADIUS - LAND_RAISE)

    # Label connected components (8-connectivity to link diagonally touching quads)
    labeled, n_comp = ndlabel(lq, structure=np.ones((3, 3)))

    results = []
    for cid in range(1, n_comp + 1):
        mask = labeled == cid
        if mask.sum() < MIN_CONTINENT_QUADS:
            continue
        tris = _mesh_from_lq(mask, Vo, Vi)
        results.append((f"continent_{cid:03d}", tris))

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

    print("\n[2/4] Building globe mesh (continents creusés)…")
    triangles = build_globe(land_map)
    print(f"  {len(triangles):,} triangles")

    print("\n[3/4] Building continent inserts (composantes connexes)…")
    inserts = build_continent_inserts(land_map)
    print(f"  {len(inserts)} continent(s) / île(s) détecté(s)")

    print("\n[4/4] Writing STL files…")
    save_stl(triangles, OUTPUT_STL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tris in inserts:
        save_stl(tris, OUTPUT_DIR / f"{name}.stl")

    print(f"\nDone.")
    print(f"  {OUTPUT_STL.name}  → globe avec continents creusés")
    print(f"  {OUTPUT_DIR}/     → {len(inserts)} fichiers STL (un par continent/île)")
