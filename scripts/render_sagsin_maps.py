#!/usr/bin/env python3
"""
Generate illustrative figures from the SAGSIN dataset tutorial scenes.

The script mirrors the Madagascar–Mozambique channel and Western North America
examples from ../sagsin-dataset/tutorial.ipynb, writes tight-bound PNGs into
assets/figures/, and disables interactive display.
"""

import datetime as dt
import random
import sys
from pathlib import Path
import importlib.util

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_REPO = BASE_DIR.parent / "sagsin-dataset"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(DATASET_REPO))

ENV_PATH = DATASET_REPO / "sagsin" / "environment.py"
spec = importlib.util.spec_from_file_location("sagsin_env", ENV_PATH)
assert spec and spec.loader
env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env)  # type: ignore


def _make_output_dir() -> Path:
    out_dir = BASE_DIR / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _configure_pm():
    pm = env.PlotManager()
    pm.fontsize = 12
    pm.markersize = 8
    return pm


def render_madagascar(out_dir: Path):
    latitude_range = [-23, -12]
    longitude_range = [32, 50]
    random_seed = 7
    random.seed(random_seed)

    map_dir = Path(env.DEFAULT_MAP_DIR)
    ground_coords = env.load_ground_positions(
        str(map_dir / "ground_madagascar_moz.csv"),
        latitude_range,
        longitude_range,
        duplicate_tol=0.2,
    )
    maritime_coords = random.sample(
        env.load_maritime_positions(str(map_dir / "maritime_positions.csv")),
        60,
    )
    haps_coords = env.load_haps_positions(
        str(map_dir / "haps_positions.csv"),
        dt.datetime(2020, 9, 28, 10, 0),
    )
    leo_coords = random.sample(
        env.load_leo_positions(
            str(map_dir / "starlink_positions.csv"),
            latitude_range,
            longitude_range,
        ),
        5,
    )

    area_size = (latitude_range[1] - latitude_range[0]) * (
        longitude_range[1] - longitude_range[0]
    )
    config = {
        "num_maritime_basestations": int(area_size / 4),
        "num_ground_basestations": int(area_size / 4),
        "num_haps_basestations": int(area_size / 50),
        "num_leo_basestations": int(area_size / 60),
        "num_users": int(area_size / 15),
        "random_seed": random_seed,
        "longitude_range": longitude_range,
        "latitude_range": latitude_range,
    }

    dm = env.DataManager(
        **config,
        ground_coords=ground_coords,
        maritime_coords=maritime_coords,
        haps_coords=haps_coords,
        leo_coords=leo_coords,
    )
    pm = _configure_pm()
    fig, _ = pm.plot(dm, legend=True, save_path="", show=False)
    fig.savefig(out_dir / "madagascar_channel.png", dpi=300, bbox_inches="tight")
    return (latitude_range, longitude_range)


def render_western_na(out_dir: Path, base_bounds):
    latitude_range = [21.3, 39.7]
    longitude_range = [-120, -90]
    random_seed = 3
    random.seed(random_seed)

    map_dir = Path(env.DEFAULT_MAP_DIR)

    ground_coords_mex = env.load_ground_positions(
        str(map_dir / "ground_mexico.csv"),
        latitude_range,
        longitude_range,
        duplicate_tol=0.3,
    )
    ground_coords_usa = env.load_ground_positions(
        str(map_dir / "ground_usa_west.csv"),
        latitude_range,
        longitude_range,
        duplicate_tol=0.3,
    )

    # sample subset for readability
    ground_coords = random.sample(
        ground_coords_mex + ground_coords_usa,
        max(1, int((len(ground_coords_mex) + len(ground_coords_usa)) * 0.2)),
    )
    maritime_coords = random.sample(
        env.load_maritime_positions(str(map_dir / "maritime_positions.csv")),
        50,
    )
    haps_coords = env.load_haps_positions(
        str(map_dir / "haps_positions.csv"),
        dt.datetime(2020, 7, 29, 0, 0),
    )
    leo_coords = env.load_leo_positions(
        str(map_dir / "starlink_positions.csv"),
        latitude_range,
        longitude_range,
    )
    user_coords = [
        (random.uniform(30, 37), random.uniform(-104, -92)) for _ in range(12)
    ]

    area_size = (latitude_range[1] - latitude_range[0]) * (
        longitude_range[1] - longitude_range[0]
    )
    config = {
        "num_maritime_basestations": int(area_size / 4),
        "num_ground_basestations": int(area_size / 4),
        "num_haps_basestations": int(area_size / 50),
        "num_leo_basestations": 1,
        "num_users": int(area_size / 15),
        "random_seed": random_seed,
        "longitude_range": longitude_range,
        "latitude_range": latitude_range,
    }

    dm = env.DataManager(
        **config,
        source_coords=[(26, -112)],
        ground_coords=ground_coords,
        maritime_coords=maritime_coords,
        haps_coords=haps_coords,
        leo_coords=leo_coords,
        user_coords=user_coords,
    )
    pm = _configure_pm()
    fig, ax = pm.plot(dm, legend=True, save_path="", show=False)
    ax.set_yticks([23, 28, 33, 38])
    fig.savefig(out_dir / "western_north_america.png", dpi=300, bbox_inches="tight")


def main():
    out_dir = _make_output_dir()
    bounds = render_madagascar(out_dir)
    render_western_na(out_dir, bounds)
    print(f"Wrote figures into {out_dir}")


if __name__ == "__main__":
    main()
