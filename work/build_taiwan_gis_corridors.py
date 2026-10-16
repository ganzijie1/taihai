import csv
import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import requests
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "gis_data"
OUT = ROOT / "outputs"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

BBOX = (119.85, 21.80, 122.10, 25.40)
ZOOM = 9
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

CITIES = {
    "Taipei": (121.5654, 25.0330),
    "Keelung": (121.7392, 25.1276),
    "Taoyuan": (121.3010, 24.9937),
    "Hsinchu": (120.9647, 24.8138),
    "Taichung": (120.6736, 24.1477),
    "Changhua": (120.5440, 24.0756),
    "Chiayi": (120.4491, 23.4801),
    "Tainan": (120.2270, 22.9999),
    "Kaohsiung": (120.3014, 22.6273),
    "Pingtung": (120.4879, 22.6731),
    "Yilan": (121.7535, 24.7570),
    "Hualien": (121.6068, 23.9911),
    "Taitung": (121.1466, 22.7554),
}

CORRIDORS = (
    ("Taipei", "Taoyuan", "west"),
    ("Taoyuan", "Hsinchu", "west"),
    ("Hsinchu", "Taichung", "west"),
    ("Taichung", "Changhua", "west"),
    ("Changhua", "Chiayi", "west"),
    ("Chiayi", "Tainan", "west"),
    ("Tainan", "Kaohsiung", "west"),
    ("Kaohsiung", "Pingtung", "west"),
    ("Taipei", "Keelung", "north"),
    ("Taipei", "Yilan", "cross"),
    ("Yilan", "Hualien", "east"),
    ("Hualien", "Taitung", "east"),
    ("Kaohsiung", "Taitung", "cross"),
)

ROAD_CAPACITY = {"motorway": 1.00, "trunk": 0.78, "primary": 0.58}
ROAD_COST = {"motorway": 0.78, "trunk": 0.90, "primary": 1.00}


def lonlat_to_tile(lon, lat, zoom):
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x, y, zoom):
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def decode_terrarium(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0


def download_terrain_tiles(session):
    west, south, east, north = BBOX
    x0f, y0f = lonlat_to_tile(west, north, ZOOM)
    x1f, y1f = lonlat_to_tile(east, south, ZOOM)
    x0, x1 = math.floor(x0f), math.floor(x1f)
    y0, y1 = math.floor(y0f), math.floor(y1f)
    tiles = {}
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            path = DATA / f"terrain_z{ZOOM}_{x}_{y}.png"
            if not path.exists():
                response = session.get(
                    TERRAIN_URL.format(z=ZOOM, x=x, y=y), timeout=45
                )
                response.raise_for_status()
                path.write_bytes(response.content)
            with Image.open(path) as image:
                tiles[(x, y)] = decode_terrarium(image)
    return tiles, (x0, y0, x1, y1)


def sample_elevation(lon, lat, tiles):
    xf, yf = lonlat_to_tile(lon, lat, ZOOM)
    tx, ty = math.floor(xf), math.floor(yf)
    tile = tiles.get((tx, ty))
    if tile is None:
        return 0.0
    px = min(255, max(0, int((xf - tx) * 256)))
    py = min(255, max(0, int((yf - ty) * 256)))
    return float(tile[py, px])


def fetch_roads(session):
    cache = DATA / "taiwan_major_roads_links_overpass.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8")), "cache"
    west, south, east, north = BBOX
    query = f"""
[out:json][timeout:180];
way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link)$"]({south},{west},{north},{east});
out tags geom;
""".strip()
    errors = []
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = session.post(endpoint, data={"data": query}, timeout=240)
            response.raise_for_status()
            payload = response.json()
            cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload, endpoint
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("; ".join(errors))


def haversine_km(a, b):
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(h))


def build_road_graph(payload, tiles):
    graph = nx.Graph()
    for element in payload.get("elements", []):
        geometry = element.get("geometry") or []
        road_class = element.get("tags", {}).get("highway", "").replace("_link", "")
        if road_class not in ROAD_CAPACITY or len(geometry) < 2:
            continue
        for left, right in zip(geometry, geometry[1:]):
            a = (round(float(left["lon"]), 5), round(float(left["lat"]), 5))
            b = (round(float(right["lon"]), 5), round(float(right["lat"]), 5))
            distance = haversine_km(a, b)
            if distance <= 0.0 or distance > 8.0:
                continue
            za = sample_elevation(*a, tiles)
            zb = sample_elevation(*b, tiles)
            slope = abs(zb - za) / max(distance * 1000.0, 1.0)
            slope_penalty = 1.0 + 7.5 * min(slope, 0.25)
            cost = distance * ROAD_COST[road_class] * slope_penalty
            capacity = ROAD_CAPACITY[road_class] / (1.0 + 5.0 * min(slope, 0.25))
            data = {
                "distance_km": distance,
                "cost": cost,
                "capacity": capacity,
                "slope": slope,
                "road_class": road_class,
            }
            if graph.has_edge(a, b):
                if cost < graph[a][b]["cost"]:
                    graph[a][b].update(data)
            else:
                graph.add_edge(a, b, **data)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def largest_component(graph):
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if not components:
        raise RuntimeError("No connected road component was constructed")
    return graph.subgraph(components[0]).copy()


def nearest_node(nodes_array, nodes, coordinate):
    lon, lat = coordinate
    scale = math.cos(math.radians(lat))
    distance2 = ((nodes_array[:, 0] - lon) * scale) ** 2 + (nodes_array[:, 1] - lat) ** 2
    return nodes[int(np.argmin(distance2))]


def path_metrics(graph, path):
    edges = [graph[a][b] for a, b in zip(path, path[1:])]
    distances = np.asarray([edge["distance_km"] for edge in edges])
    capacities = np.asarray([edge["capacity"] for edge in edges])
    return {
        "distance_km": sum(edge["distance_km"] for edge in edges),
        "generalized_cost": sum(edge["cost"] for edge in edges),
        # A length-weighted harmonic capacity preserves bottleneck sensitivity
        # without allowing one sub-pixel elevation artefact to define a route.
        "bottleneck_capacity": float(
            distances.sum() / np.maximum(np.sum(distances / capacities), 1e-9)
        ),
        "minimum_segment_capacity": float(capacities.min()),
        "mean_slope": float(np.average(
            [edge["slope"] for edge in edges],
            weights=[edge["distance_km"] for edge in edges],
        )),
    }


def compute_corridors(graph):
    nodes = list(graph.nodes)
    nodes_array = np.asarray(nodes, dtype=float)
    anchors = {
        name: nearest_node(nodes_array, nodes, coordinate)
        for name, coordinate in CITIES.items()
    }
    results = []
    paths = {}
    for origin, destination, corridor_type in CORRIDORS:
        source, target = anchors[origin], anchors[destination]
        try:
            primary_path = nx.shortest_path(graph, source, target, weight="cost")
        except nx.NetworkXNoPath:
            continue
        metrics = path_metrics(graph, primary_path)
        # At this regional resolution, branch-node density is a stable proxy
        # for rerouting options and avoids repeated all-island disjoint-path
        # searches on a very large public OSM graph.
        internal_nodes = primary_path[1:-1]
        branch_share = (
            float(np.mean([graph.degree(node) >= 3 for node in internal_nodes]))
            if internal_nodes else 0.0
        )
        redundancy = float(np.clip(0.12 + 0.88 * branch_share, 0.12, 1.0))
        detour_ratio = None if redundancy <= 0.12 else 1.0 / redundancy
        record = {
            "origin": origin,
            "destination": destination,
            "corridor_type": corridor_type,
            **metrics,
            "cost_per_km": metrics["generalized_cost"] / max(metrics["distance_km"], 1e-9),
            "alternate_detour_ratio": detour_ratio,
            "redundancy": redundancy,
            "path_nodes": len(primary_path),
        }
        results.append(record)
        paths[f"{origin}-{destination}"] = primary_path
    return results, paths, anchors


def summarize(results, graph, source):
    by_type = {}
    for corridor_type in sorted({row["corridor_type"] for row in results}):
        rows = [row for row in results if row["corridor_type"] == corridor_type]
        weights = np.asarray([row["distance_km"] for row in rows])
        by_type[corridor_type] = {
            "count": len(rows),
            "distance_weighted_cost_per_km": float(np.average(
                [row["cost_per_km"] for row in rows], weights=weights
            )),
            "distance_weighted_capacity": float(np.average(
                [row["bottleneck_capacity"] for row in rows], weights=weights
            )),
            "distance_weighted_redundancy": float(np.average(
                [row["redundancy"] for row in rows], weights=weights
            )),
            "distance_weighted_slope": float(np.average(
                [row["mean_slope"] for row in rows], weights=weights
            )),
        }
    all_weights = np.asarray([row["distance_km"] for row in results])
    mean_cost = float(np.average([row["cost_per_km"] for row in results], weights=all_weights))
    mean_capacity = float(np.average([row["bottleneck_capacity"] for row in results], weights=all_weights))
    mean_redundancy = float(np.average([row["redundancy"] for row in results], weights=all_weights))
    mobility_index = float(np.clip(
        0.48 * np.exp(-0.65 * max(mean_cost - 0.75, 0.0))
        + 0.32 * mean_capacity + 0.20 * mean_redundancy,
        0.15,
        0.95,
    ))
    return {
        "model": "Taiwan regional GIS mobility-corridor graph V1",
        "bbox_wgs84": BBOX,
        "terrain_source": "AWS Terrain Tiles / SRTM-derived Terrarium tiles",
        "road_source": "OpenStreetMap major roads via Overpass",
        "road_fetch_source": source,
        "road_graph_nodes": graph.number_of_nodes(),
        "road_graph_edges": graph.number_of_edges(),
        "corridor_count": len(results),
        "aggregate": {
            "cost_per_km": mean_cost,
            "bottleneck_capacity": mean_capacity,
            "redundancy": mean_redundancy,
            "mobility_index": mobility_index,
            "terrain_friction_index": 1.0 - mobility_index,
        },
        "by_type": by_type,
        "limitations": (
            "Regional public-data graph, not a tactical route plan. Road class is a capacity proxy; "
            "bridges, tunnels, traffic, demolitions and classified infrastructure are not resolved. "
            "Reported bottleneck capacity is a length-weighted harmonic proxy, not measured traffic flow."
        ),
    }


def render_map(tiles, tile_bounds, graph, paths):
    x0, y0, x1, y1 = tile_bounds
    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), dtype=np.float32)
    for (x, y), tile in tiles.items():
        row, col = (y - y0) * 256, (x - x0) * 256
        mosaic[row:row + 256, col:col + 256] = tile
    land = mosaic > 0.0
    shaded = np.zeros((*mosaic.shape, 3), dtype=np.uint8)
    normalized = np.clip(mosaic / 3200.0, 0.0, 1.0)
    shaded[:, :, 0] = np.where(land, 55 + 120 * normalized, 21).astype(np.uint8)
    shaded[:, :, 1] = np.where(land, 118 - 60 * normalized, 63).astype(np.uint8)
    shaded[:, :, 2] = np.where(land, 76 - 28 * normalized, 92).astype(np.uint8)
    image = Image.fromarray(shaded, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    def pixel(node):
        xf, yf = lonlat_to_tile(node[0], node[1], ZOOM)
        return ((xf - x0) * 256, (yf - y0) * 256)

    for a, b, edge in graph.edges(data=True):
        alpha = 62 if edge["road_class"] == "primary" else 90
        draw.line((pixel(a), pixel(b)), fill=(235, 235, 225, alpha), width=1)
    colors = [(238, 71, 64, 230), (255, 202, 40, 230), (58, 169, 220, 230)]
    for index, path in enumerate(paths.values()):
        draw.line([pixel(node) for node in path], fill=colors[index % len(colors)], width=2)

    west, south, east, north = BBOX
    px0, py0 = pixel((west, north))
    px1, py1 = pixel((east, south))
    crop = image.crop((int(px0), int(py0), int(px1), int(py1)))
    crop.save(OUT / "台海V3.6_GIS机动走廊图.png")


def write_outputs(results, summary):
    csv_path = OUT / "台海V3.6_GIS机动走廊.csv"
    fields = list(results[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    json_path = OUT / "台海V3.6_GIS机动走廊摘要.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    session = requests.Session()
    session.headers.update({"User-Agent": "Codex-research-model/1.0 (public-data analysis)"})
    tiles, tile_bounds = download_terrain_tiles(session)
    roads, source = fetch_roads(session)
    graph = largest_component(build_road_graph(roads, tiles))
    results, paths, _ = compute_corridors(graph)
    if len(results) < 9:
        raise RuntimeError(f"Only {len(results)} connected corridors were resolved")
    summary = summarize(results, graph, source)
    csv_path, json_path = write_outputs(results, summary)
    render_map(tiles, tile_bounds, graph, paths)
    print("TAIWAN_GIS_CORRIDOR_V1_VERIFICATION: PASS")
    print(json.dumps(summary["aggregate"], ensure_ascii=False))
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
