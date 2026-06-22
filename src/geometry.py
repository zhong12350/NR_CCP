"""Field geometry: WKT polygons, headland, local metric projection, raster grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import affinity, wkt
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.prepared import prep

EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class FieldGeometry:
    """Farmland polygon with headland ring and inner workable area."""

    name: str
    outer: Polygon
    inner: Polygon
    headland: Polygon
    crs: str = "metric"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.outer.bounds

    @property
    def area_m2(self) -> float:
        return float(self.outer.area)

    @property
    def aspect_ratio(self) -> float:
        minx, miny, maxx, maxy = self.bounds
        w = max(maxx - minx, 1e-6)
        h = max(maxy - miny, 1e-6)
        return max(w, h) / min(w, h)


@dataclass(frozen=True)
class FieldGrid:
    """Rasterized field aligned with outer polygon bounds."""

    geometry: FieldGeometry
    cell_size_m: float
    origin_x: float
    origin_y: float
    outer_mask: np.ndarray
    inner_mask: np.ndarray
    headland_mask: np.ndarray

    @property
    def nx(self) -> int:
        return int(self.outer_mask.shape[1])

    @property
    def ny(self) -> int:
        return int(self.outer_mask.shape[0])

    @property
    def x_coords(self) -> np.ndarray:
        return self.origin_x + (np.arange(self.nx) + 0.5) * self.cell_size_m

    @property
    def y_coords(self) -> np.ndarray:
        return self.origin_y + (np.arange(self.ny) + 0.5) * self.cell_size_m

    def world_to_index(self, x: float, y: float) -> tuple[int, int]:
        ix = int((x - self.origin_x) / self.cell_size_m)
        iy = int((y - self.origin_y) / self.cell_size_m)
        ix = int(np.clip(ix, 0, self.nx - 1))
        iy = int(np.clip(iy, 0, self.ny - 1))
        return ix, iy

    def index_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        return float(self.x_coords[ix]), float(self.y_coords[iy])

    def contains_inner(self, x: float, y: float) -> bool:
        ix, iy = self.world_to_index(x, y)
        return bool(self.inner_mask[iy, ix])

    def contains_headland(self, x: float, y: float) -> bool:
        ix, iy = self.world_to_index(x, y)
        return bool(self.headland_mask[iy, ix])


def is_geographic(polygon: Polygon) -> bool:
    """Heuristic: Fields2Benchmark WKT uses lon/lat degrees."""
    minx, miny, maxx, maxy = polygon.bounds
    return (
        abs(minx) <= 180.0
        and abs(maxx) <= 180.0
        and abs(miny) <= 90.0
        and abs(maxy) <= 90.0
        and (maxx - minx) < 1.0
        and (maxy - miny) < 1.0
    )


def project_polygon_to_local_meters(polygon: Polygon) -> Polygon:
    """Convert WGS84 lon/lat polygon to local equirectangular meters."""
    lon0, lat0 = polygon.centroid.x, polygon.centroid.y
    cos_lat = float(np.cos(np.deg2rad(lat0)))

    def _fwd(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        xm = np.deg2rad(x - lon0) * cos_lat * EARTH_RADIUS_M
        ym = np.deg2rad(y - lat0) * EARTH_RADIUS_M
        return xm, ym

    projected = transform(_fwd, polygon)
    if not projected.is_valid:
        projected = projected.buffer(0)
    return projected


def load_wkt_polygon(path: str | Path, auto_project: bool = True) -> tuple[Polygon, str]:
    """Load polygon from WKT; optionally project geographic coords to meters."""
    text = Path(path).read_text(encoding="utf-8").strip()
    geom = wkt.loads(text)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon):
        raise ValueError(f"Expected Polygon WKT, got {geom.geom_type}")
    if not geom.is_valid:
        geom = geom.buffer(0)

    crs = "metric"
    if auto_project and is_geographic(geom):
        geom = project_polygon_to_local_meters(geom)
        crs = "local_meters"
    return geom, crs


def suggest_cell_size(polygon: Polygon, target_cells_long_side: int = 80) -> float:
    """Pick raster cell size from field extent."""
    minx, miny, maxx, maxy = polygon.bounds
    extent = max(maxx - minx, maxy - miny, 1.0)
    return float(np.clip(extent / target_cells_long_side, 1.0, 8.0))


def build_field_geometry(
    polygon: Polygon,
    headland_width_m: float,
    name: str = "field",
    crs: str = "metric",
) -> FieldGeometry:
    """Create outer / inner / headland regions."""
    outer = polygon
    inner = outer.buffer(-headland_width_m)
    if inner.is_empty or inner.area < 1.0:
        shrink = max(headland_width_m * 0.35, 1.0)
        inner = outer.buffer(-shrink)
    if inner.is_empty:
        inner = outer.buffer(-1.0)
    if inner.geom_type == "MultiPolygon":
        inner = max(inner.geoms, key=lambda g: g.area)
    headland = outer.difference(inner)
    return FieldGeometry(name=name, outer=outer, inner=inner, headland=headland, crs=crs)


def rasterize_geometry(
    geometry: FieldGeometry,
    cell_size_m: float,
    padding_m: float = 2.0,
) -> FieldGrid:
    """Rasterize outer, inner, and headland masks."""
    minx, miny, maxx, maxy = geometry.bounds
    minx -= padding_m
    miny -= padding_m
    maxx += padding_m
    maxy += padding_m

    nx = max(1, int(np.ceil((maxx - minx) / cell_size_m)))
    ny = max(1, int(np.ceil((maxy - miny) / cell_size_m)))

    outer_mask = np.zeros((ny, nx), dtype=bool)
    inner_mask = np.zeros((ny, nx), dtype=bool)
    headland_mask = np.zeros((ny, nx), dtype=bool)

    outer_prep = prep(geometry.outer)
    inner_prep = prep(geometry.inner)
    headland_prep = prep(geometry.headland)

    for iy in range(ny):
        y = miny + (iy + 0.5) * cell_size_m
        for ix in range(nx):
            x = minx + (ix + 0.5) * cell_size_m
            pt = Point(x, y)
            if outer_prep.contains(pt) or geometry.outer.touches(pt):
                outer_mask[iy, ix] = True
            if inner_prep.contains(pt):
                inner_mask[iy, ix] = True
            if headland_prep.contains(pt):
                headland_mask[iy, ix] = True

    return FieldGrid(
        geometry=geometry,
        cell_size_m=cell_size_m,
        origin_x=minx,
        origin_y=miny,
        outer_mask=outer_mask,
        inner_mask=inner_mask,
        headland_mask=headland_mask,
    )


def load_field_from_wkt(
    wkt_path: str | Path,
    headland_width_m: float,
    cell_size_m: float | None = None,
    auto_cell_size: bool = False,
) -> FieldGrid:
    """Load WKT polygon and build rasterized field grid in local meters."""
    path = Path(wkt_path)
    polygon, crs = load_wkt_polygon(path)
    if auto_cell_size or cell_size_m is None:
        cell_size_m = suggest_cell_size(polygon)
    geometry = build_field_geometry(polygon, headland_width_m, name=path.stem, crs=crs)
    return rasterize_geometry(geometry, float(cell_size_m))


def rotate_geometry(geom: BaseGeometry, angle_deg: float, origin: tuple[float, float]):
    """Rotate geometry about origin (centroid by default)."""
    ox, oy = origin
    return affinity.rotate(geom, angle_deg, origin=(ox, oy), use_radians=False)


def sample_line(
    line: LineString,
    spacing_m: float,
) -> list[tuple[float, float]]:
    """Sample points along a linestring at approximately fixed spacing."""
    if line.is_empty or line.length < 1e-6:
        return []
    n = max(2, int(np.ceil(line.length / spacing_m)) + 1)
    distances = np.linspace(0.0, line.length, n)
    return [(float(line.interpolate(d).x), float(line.interpolate(d).y)) for d in distances]


def extract_linestrings(geom: BaseGeometry) -> list[LineString]:
    """Normalize intersection results to a list of LineStrings."""
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if g.length > 1e-6]
    if geom.geom_type == "GeometryCollection":
        lines: list[LineString] = []
        for g in geom.geoms:
            lines.extend(extract_linestrings(g))
        return lines
    return []
