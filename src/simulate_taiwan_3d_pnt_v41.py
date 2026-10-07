import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_taiwan_gis_corridors import (
    BBOX,
    ZOOM,
    download_terrain_tiles,
    lonlat_to_tile,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
V40_PATH = OUT / "台海V4.0双方对称相关失效与稀有事件_仿真摘要.json"
SUMMARY_PATH = OUT / "台海V4.1三维地形与多源PNT_仿真摘要.json"
CSV_PATH = OUT / "台海V4.1三维地形与多源PNT_条件结果.csv"
CONTOUR_PATH = OUT / "台海V4.1_DEM等高线与天空可见度.png"
PERSPECTIVE_PATH = OUT / "台海V4.1_三维地形透视图.png"
POINT_CLOUD_PATH = OUT / "台海V4.1_DEM降采样点云.ply"

PATHS = 30000
MONTHS = 60
SEED = 20261004
SIDES = ("china", "united_states", "japan", "taiwan")
SCENARIOS = {
    "baseline_contested": {
        "label": "基准竞争环境",
        "space": (0.90, 0.91, 0.90, 0.89),
        "ground": (0.93, 0.94, 0.94, 0.93),
        "common": 0.02,
    },
    "regional_interference": {
        "label": "区域性干扰与欺骗",
        "space": (0.72, 0.66, 0.65, 0.58),
        "ground": (0.87, 0.88, 0.87, 0.84),
        "common": 0.08,
    },
    "symmetric_gnss_denial": {
        "label": "双方卫星导航显著受限",
        "space": (0.43, 0.42, 0.44, 0.38),
        "ground": (0.79, 0.80, 0.80, 0.76),
        "common": 0.13,
    },
    "external_pnt_disruption": {
        "label": "美日台PNT链路偏重受损",
        "space": (0.72, 0.39, 0.41, 0.35),
        "ground": (0.88, 0.68, 0.63, 0.58),
        "common": 0.09,
    },
    "china_pnt_disruption": {
        "label": "中国PNT链路偏重受损",
        "space": (0.39, 0.72, 0.71, 0.64),
        "ground": (0.61, 0.87, 0.86, 0.82),
        "common": 0.09,
    },
    "common_mode_space_event": {
        "label": "共同空间环境冲击",
        "space": (0.50, 0.50, 0.51, 0.47),
        "ground": (0.84, 0.85, 0.85, 0.82),
        "common": 0.22,
    },
}

# These are transparent structural priors, not classified performance estimates.
# Japan's row represents GPS/QZSS fusion; Taiwan's represents public multi-GNSS
# and terrestrial control infrastructure, not a sovereign constellation.
ACTOR_PRIORS = {
    "china": {
        "space": 0.91, "ground": 0.88, "alternative": 0.82,
        "integrity": 0.84, "map": 0.88, "protected": 0.88,
    },
    "united_states": {
        "space": 0.94, "ground": 0.78, "alternative": 0.91,
        "integrity": 0.92, "map": 0.90, "protected": 0.96,
    },
    "japan": {
        "space": 0.86, "ground": 0.91, "alternative": 0.82,
        "integrity": 0.91, "map": 0.92, "protected": 0.78,
    },
    "taiwan": {
        "space": 0.72, "ground": 0.90, "alternative": 0.70,
        "integrity": 0.78, "map": 0.91, "protected": 0.45,
    },
}

CASE_WEIGHTS = {
    "timely_full": np.array([0.50, 0.27, 0.23]),
    "limited": np.array([0.23, 0.22, 0.55]),
    "delayed": np.array([0.31, 0.24, 0.45]),
    "japan_only": np.array([0.00, 0.46, 0.54]),
    "taiwan_alone": np.array([0.00, 0.00, 1.00]),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logit(p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def reconstruct_dem():
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "Codex-research-model/1.0 (public-data analysis)"})
    tiles, (x0, y0, x1, y1) = download_terrain_tiles(session)
    mosaic = np.full(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.nan, dtype=np.float32)
    for (x, y), tile in tiles.items():
        row, col = (y - y0) * 256, (x - x0) * 256
        mosaic[row:row + 256, col:col + 256] = tile

    west, south, east, north = BBOX
    xf0, yf0 = lonlat_to_tile(west, north, ZOOM)
    xf1, yf1 = lonlat_to_tile(east, south, ZOOM)
    left = int(round((xf0 - x0) * 256))
    right = int(round((xf1 - x0) * 256))
    top = int(round((yf0 - y0) * 256))
    bottom = int(round((yf1 - y0) * 256))
    return mosaic[top:bottom, left:right]


def terrain_metrics(dem):
    land = dem > 0.0
    height = np.where(land, dem, np.nan)
    rows, cols = dem.shape
    mid_lat = 0.5 * (BBOX[1] + BBOX[3])
    dx = 111320.0 * math.cos(math.radians(mid_lat)) * (BBOX[2] - BBOX[0]) / cols
    dy = 111320.0 * (BBOX[3] - BBOX[1]) / rows
    filled = np.where(land, dem, 0.0)
    gy, gx = np.gradient(filled, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    slope = np.where(land, slope, np.nan)

    offsets = (-8, -4, 4, 8)
    samples = [filled]
    for offset in offsets:
        samples.append(np.roll(filled, offset, axis=0))
        samples.append(np.roll(filled, offset, axis=1))
    local_relief = np.nanstd(np.stack(samples), axis=0)
    local_relief = np.where(land, local_relief, np.nan)
    rough = np.clip(local_relief / 650.0, 0.0, 1.0)
    sky_view = np.clip(np.exp(-1.9 * np.nan_to_num(slope)) - 0.14 * np.nan_to_num(rough), 0.25, 0.99)
    sky_view = np.where(land, sky_view, np.nan)
    terrain_observability = np.clip(0.20 + 0.52 * rough + 0.34 * np.sin(np.nan_to_num(slope)), 0.15, 0.96)
    terrain_observability = np.where(land, terrain_observability, np.nan)

    values = height[land]
    slope_values = np.degrees(slope[land])
    result = {
        "grid_shape": [int(rows), int(cols)],
        "approx_grid_spacing_m": [float(dx), float(dy)],
        "elevation_m_p10_p50_p90_p99": [float(x) for x in np.quantile(values, [0.10, 0.50, 0.90, 0.99])],
        "maximum_elevation_m_in_grid": float(np.max(values)),
        "slope_degrees_p10_p50_p90": [float(x) for x in np.quantile(slope_values, [0.10, 0.50, 0.90])],
        "land_share_above_500m": float(np.mean(values > 500.0)),
        "land_share_above_1000m": float(np.mean(values > 1000.0)),
        "sky_view_proxy_p10_p50_p90": [float(x) for x in np.quantile(sky_view[land], [0.10, 0.50, 0.90])],
        "terrain_observability_p10_p50_p90": [float(x) for x in np.quantile(terrain_observability[land], [0.10, 0.50, 0.90])],
        "data_scope": "regional DEM; DSM and raw LiDAR point clouds are interface-only because unrestricted high-resolution coverage was not ingested",
    }
    arrays = {
        "land": land,
        "slope": slope,
        "sky_view": sky_view,
        "terrain_observability": terrain_observability,
    }
    return result, arrays


def color_for_height(z, shade=1.0):
    if z <= 0:
        rgb = (26, 70, 103)
    elif z < 250:
        t = z / 250.0
        rgb = (64 + 38 * t, 137 + 34 * t, 79 - 18 * t)
    elif z < 1200:
        t = (z - 250.0) / 950.0
        rgb = (102 + 82 * t, 171 - 60 * t, 61 - 24 * t)
    elif z < 2800:
        t = (z - 1200.0) / 1600.0
        rgb = (184 - 28 * t, 111 - 35 * t, 37 + 24 * t)
    else:
        t = min((z - 2800.0) / 1200.0, 1.0)
        rgb = (156 + 86 * t, 76 + 166 * t, 61 + 181 * t)
    return tuple(int(np.clip(v * shade, 0, 255)) for v in rgb)


def render_contours(dem, arrays):
    land = arrays["land"]
    h, w = dem.shape
    scale = min(1400.0 / w, 1000.0 / h)
    out_w, out_h = int(w * scale), int(h * scale)
    image = Image.new("RGB", (w, h), (24, 62, 92))
    pixels = image.load()
    slope = arrays["slope"]
    sky = arrays["sky_view"]
    for y in range(h):
        for x in range(w):
            if not land[y, x]:
                continue
            shade = 0.68 + 0.32 * float(sky[y, x])
            pixels[x, y] = color_for_height(float(dem[y, x]), shade)
    quant = np.floor(np.maximum(dem, 0.0) / 250.0)
    edge = land & (
        (quant != np.roll(quant, 1, axis=0))
        | (quant != np.roll(quant, 1, axis=1))
    )
    for y, x in np.argwhere(edge):
        pixels[int(x), int(y)] = (235, 238, 225)
    image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 540, 84), fill=(10, 22, 31, 205))
    draw.text((32, 30), "Taiwan regional DEM / 250 m contours", fill=(245, 245, 240))
    draw.text((32, 54), "brightness includes terrain sky-view proxy", fill=(210, 218, 220))
    image.save(CONTOUR_PATH)


def render_perspective(dem, arrays):
    step = max(4, int(max(dem.shape) / 135))
    z = dem[::step, ::step]
    land = arrays["land"][::step, ::step]
    nr, nc = z.shape
    canvas = Image.new("RGB", (1500, 920), (18, 42, 61))
    draw = ImageDraw.Draw(canvas)

    def project(row, col, elev):
        x = 750 + (col - nc / 2.0) * 6.0 - (row - nr / 2.0) * 2.1
        y = 560 + (row - nr / 2.0) * 3.0 - elev * 0.075
        return x, y

    for row in range(nr - 2, -1, -1):
        for col in range(nc - 1):
            if not (land[row, col] or land[row + 1, col] or land[row, col + 1] or land[row + 1, col + 1]):
                continue
            points = [
                project(row, col, max(z[row, col], 0.0)),
                project(row, col + 1, max(z[row, col + 1], 0.0)),
                project(row + 1, col + 1, max(z[row + 1, col + 1], 0.0)),
                project(row + 1, col, max(z[row + 1, col], 0.0)),
            ]
            mean_z = float(np.mean([max(z[row, col], 0.0), max(z[row, col + 1], 0.0), max(z[row + 1, col], 0.0), max(z[row + 1, col + 1], 0.0)]))
            draw.polygon(points, fill=color_for_height(mean_z, 0.92), outline=(28, 49, 44))
    draw.rectangle((26, 24, 600, 96), fill=(8, 18, 26))
    draw.text((42, 38), "Taiwan regional 3D terrain surface", fill=(248, 248, 242))
    draw.text((42, 64), "regional DEM, vertical exaggeration for inspection", fill=(204, 216, 218))
    canvas.save(PERSPECTIVE_PATH)


def write_point_cloud(dem, arrays):
    land = arrays["land"]
    stride = max(4, int(max(dem.shape) / 180))
    points = []
    rows, cols = dem.shape
    for row in range(0, rows, stride):
        lat = BBOX[3] - (row + 0.5) / rows * (BBOX[3] - BBOX[1])
        for col in range(0, cols, stride):
            if not land[row, col]:
                continue
            lon = BBOX[0] + (col + 0.5) / cols * (BBOX[2] - BBOX[0])
            points.append((lon, lat, float(dem[row, col])))
    header = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property double longitude", "property double latitude", "property float elevation_m",
        "end_header",
    ]
    with POINT_CLOUD_PATH.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(header) + "\n")
        for lon, lat, elevation in points:
            handle.write(f"{lon:.7f} {lat:.7f} {elevation:.2f}\n")
    return len(points)


def evolve_pnt(rng, actor, scenario, terrain, n):
    prior = ACTOR_PRIORS[actor]
    actor_index = SIDES.index(actor)
    space = np.clip(rng.normal(prior["space"], 0.035, n), 0.05, 0.995)
    ground = np.clip(rng.normal(prior["ground"], 0.045, n), 0.05, 0.995)
    alternative = np.clip(rng.normal(prior["alternative"], 0.045, n), 0.05, 0.995)
    integrity = np.clip(rng.normal(prior["integrity"], 0.04, n), 0.05, 0.995)
    map_quality = np.clip(rng.normal(prior["map"], 0.035, n), 0.05, 0.995)
    protected = prior["protected"]
    sky_p10, sky_p50, sky_p90 = terrain["sky_view_proxy_p10_p50_p90"]
    obs_p10, obs_p50, obs_p90 = terrain["terrain_observability_p10_p50_p90"]
    sky = np.clip(rng.triangular(sky_p10, sky_p50, sky_p90, n), 0.20, 0.995)
    observability = np.clip(rng.triangular(obs_p10, obs_p50, obs_p90, n), 0.10, 0.995)
    qualities = np.zeros((MONTHS, n), dtype=np.float32)
    integrity_risk = np.zeros((MONTHS, n), dtype=np.float32)

    for month in range(MONTHS):
        common_event = rng.random(n) < scenario["common"] / 12.0
        local_event = rng.random(n) < (0.08 + 0.08 * (1.0 - protected)) / 12.0
        space_target = prior["space"] * scenario["space"][actor_index]
        ground_target = prior["ground"] * scenario["ground"][actor_index]
        space += 0.22 * (space_target - space) + rng.normal(0.0, 0.018, n)
        ground += 0.18 * (ground_target - ground) + rng.normal(0.0, 0.016, n)
        alternative += 0.10 * (prior["alternative"] - alternative) + rng.normal(0.0, 0.010, n)
        space *= np.where(common_event, rng.uniform(0.52, 0.78, n), 1.0)
        space *= np.where(local_event, rng.uniform(0.45, 0.78, n), 1.0)
        ground *= np.where(local_event, rng.uniform(0.62, 0.86, n), 1.0)
        space = np.clip(space, 0.03, 0.995)
        ground = np.clip(ground, 0.03, 0.995)
        alternative = np.clip(alternative, 0.05, 0.995)

        geometry_information = 1.55 * space * sky * (0.55 + 0.45 * protected)
        control_information = 0.90 * ground * sky
        terrain_information = 0.78 * alternative * map_quality * observability
        inertial_information = 0.62 * alternative
        information = geometry_information + control_information + terrain_information + inertial_information
        deceptive = (local_event | common_event) * (1.0 - integrity)
        quality = (1.0 - np.exp(-information / 2.35)) * (1.0 - 0.42 * deceptive)
        qualities[month] = np.clip(quality, 0.03, 0.995)
        integrity_risk[month] = np.clip((1.0 - integrity) * (1.0 - space) + 0.65 * deceptive, 0.0, 1.0)
    return {
        "quality": qualities.mean(axis=0),
        "terminal_quality": qualities[-1],
        "integrity_risk": integrity_risk.mean(axis=0),
    }


def simulate_outcomes(v40, terrain):
    rng = np.random.default_rng(SEED)
    base_rows = {(row["conflict_year"], row["case"]): row for row in v40["results"]}
    pnt = {}
    for scenario_key, scenario in SCENARIOS.items():
        pnt[scenario_key] = {
            actor: evolve_pnt(rng, actor, scenario, terrain, PATHS)
            for actor in SIDES
        }

    baseline_rel = {}
    for case, weights in CASE_WEIGHTS.items():
        external = sum(
            weights[i] * pnt["baseline_contested"][actor]["quality"]
            for i, actor in enumerate(("united_states", "japan", "taiwan"))
        )
        baseline_rel[case] = pnt["baseline_contested"]["china"]["quality"] - external

    results = []
    for (year, case), base in sorted(base_rows.items()):
        base_probability = float(base["v40_importance_weighted_probability"])
        weights = CASE_WEIGHTS[case]
        for scenario_key, scenario in SCENARIOS.items():
            china = pnt[scenario_key]["china"]["quality"]
            external = sum(
                weights[i] * pnt[scenario_key][actor]["quality"]
                for i, actor in enumerate(("united_states", "japan", "taiwan"))
            )
            relative = china - external
            delta = 1.35 * (relative - baseline_rel[case])
            # Zero-mean unresolved geometry uncertainty widens the distribution
            # without changing the baseline scenario's central calibration.
            unresolved = rng.normal(0.0, 0.22, PATHS)
            raw = sigmoid(logit(base_probability) + delta + unresolved)
            if scenario_key == "baseline_contested":
                correction = logit(base_probability) - logit(float(np.mean(raw)))
                raw = sigmoid(logit(raw) + correction)
            results.append({
                "conflict_year": int(year),
                "case": case,
                "scenario": scenario_key,
                "scenario_label": scenario["label"],
                "v40_probability": base_probability,
                "v41_probability": float(np.mean(raw)),
                "v41_probability_p10_p50_p90": [float(x) for x in np.quantile(raw, [0.10, 0.50, 0.90])],
                "china_pnt_quality_p10_p50_p90": [float(x) for x in np.quantile(china, [0.10, 0.50, 0.90])],
                "external_effective_pnt_quality_p10_p50_p90": [float(x) for x in np.quantile(external, [0.10, 0.50, 0.90])],
                "relative_pnt_advantage_p10_p50_p90": [float(x) for x in np.quantile(relative, [0.10, 0.50, 0.90])],
            })
    diagnostics = {}
    for scenario_key in SCENARIOS:
        diagnostics[scenario_key] = {}
        for actor in SIDES:
            state = pnt[scenario_key][actor]
            diagnostics[scenario_key][actor] = {
                "mean_quality": float(np.mean(state["quality"])),
                "terminal_quality_p10_p50_p90": [float(x) for x in np.quantile(state["terminal_quality"], [0.10, 0.50, 0.90])],
                "mean_integrity_risk": float(np.mean(state["integrity_risk"])),
            }
    return results, diagnostics


def validate(terrain, results):
    errors = []
    if not (0.0 < terrain["sky_view_proxy_p10_p50_p90"][0] <= terrain["sky_view_proxy_p10_p50_p90"][2] <= 1.0):
        errors.append("sky-view proxy range")
    if not (0.0 < terrain["terrain_observability_p10_p50_p90"][0] <= terrain["terrain_observability_p10_p50_p90"][2] <= 1.0):
        errors.append("terrain-observability range")
    if any(not (0.0 <= row["v41_probability"] <= 1.0) for row in results):
        errors.append("probability range")
    for year in (2030, 2050, 2075, 2100):
        for case in CASE_WEIGHTS:
            rows = [r for r in results if r["conflict_year"] == year and r["case"] == case]
            if len(rows) != len(SCENARIOS):
                errors.append(f"missing cell {year} {case}")
    if errors:
        raise RuntimeError("; ".join(errors))


def write_csv(results):
    fields = [
        "conflict_year", "case", "scenario", "scenario_label",
        "v40_probability", "v41_probability",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fields})


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    dem = reconstruct_dem()
    terrain, arrays = terrain_metrics(dem)
    render_contours(dem, arrays)
    render_perspective(dem, arrays)
    terrain["point_cloud_vertices"] = write_point_cloud(dem, arrays)
    v40 = json.loads(V40_PATH.read_text(encoding="utf-8"))
    results, diagnostics = simulate_outcomes(v40, terrain)
    validate(terrain, results)
    payload = {
        "model": "V4.1 regional 3D terrain, geodetic control network and resilient multi-source PNT overlay",
        "paths_per_scenario": PATHS,
        "months": MONTHS,
        "terrain": terrain,
        "actor_priors": ACTOR_PRIORS,
        "scenario_definitions": SCENARIOS,
        "pnt_diagnostics": diagnostics,
        "results": results,
        "identification_warning": (
            "Actor parameters are transparent public-information structural priors, not classified estimates. "
            "The regional DEM is used directly; high-resolution DSM and raw LiDAR point clouds were not ingested. "
            "Outcome changes are sensitivity results conditional on V4.0, not operational forecasts."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results)
    print("TAIWAN_3D_PNT_V41_VERIFICATION: PASS")
    print(json.dumps(terrain, ensure_ascii=False))
    for row in results:
        if row["conflict_year"] == 2030 and row["case"] == "timely_full":
            print(row["scenario"], f"{100 * row['v41_probability']:.4f}%")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
