import csv
import logging
import struct
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from collections.abc import Iterable, Sequence
from fractions import Fraction
from importlib import resources
from operator import itemgetter
from pathlib import Path
from typing import Any, DefaultDict

import geopandas as gpd
import piexif
from genutility.rich import Progress
from pi_heif import register_heif_opener
from PIL import Image, UnidentifiedImageError
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn
from rich.progress import Progress as RichProgress
from shapely.geometry import Point

register_heif_opener()


class TaiwanGeoLocation:
    def __init__(self):
        gpd.options.io_engine = "pyogrio"
        VILLAGE_FILE = "data_village/VILLAGE_NLSC_1140825.shp"
        shp_resource = resources.files(__package__).joinpath(VILLAGE_FILE)

        with resources.as_file(shp_resource) as shp_path:
            gdf = gpd.read_file(shp_path)
            assert gdf.crs.name == "GCS_TWD97[2020]", gdf.crs.name

            self.gdf = gdf.to_crs(epsg=4326)

    def lookup(self, lat: float, lon: float) -> dict[str, str | None]:
        """
        Given a WGS84 coordinate (lon, lat),
        return a dict with county / town / village
        """
        pt = Point(lon, lat)

        # 1) Narrow down with spatial index
        candidate_idxs = list(self.gdf.sindex.intersection(pt.bounds))
        if not candidate_idxs:
            raise KeyError(f"Coordinates {lat}, {lon} could not be found in Taiwan")

        candidates = self.gdf.iloc[candidate_idxs]

        # 2) Exact point-in-polygon test
        hit = candidates[candidates.contains(pt)]
        if hit.empty:
            # Fallback to intersects so boundary points are included
            hit = candidates[candidates.intersects(pt)]
            if hit.empty:
                raise KeyError(f"Coordinates {lat}, {lon} could not be found in Taiwan")

        row = hit.iloc[0]

        return {
            "level1_chinese": row["COUNTYNAME"],
            "level1_english": row.get("COUNTYENG"),
            "level2_chinese": row["TOWNNAME"],
            "level2_english": row.get("TOWNENG"),
            "level3_chinese": row["VILLNAME"],
            "level3_english": row.get("VILLENG"),
        }

    def plot(self, locations: Iterable[tuple[str, float, float, str]]):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        self.gdf.plot(ax=ax, color="white", edgecolor="black")

        if locations:
            x, y = zip(*((c, b) for a, b, c, d in locations), strict=True)
            ax.scatter(x, y, marker="o", color="red")

        plt.show()


def test_loopup():
    locations = [
        ("Taipei 101", 25.033976, 121.5645389, "西村里"),
        ("Wulai Village", 24.853706, 121.5541264, "烏來里"),
        ("Shifen Village", 25.0444566, 121.767985, "十分里"),
        ("Wude Martial Arts Center", 22.624641, 120.2724156, "惠安里"),
    ]

    geoloc = TaiwanGeoLocation()
    # geoloc.plot(locations)

    for _name, lat, lon, village_zho in locations:
        d = geoloc.lookup(lat, lon)
        assert d["level3_chinese"] == village_zho, (d["level3_chinese"], village_zho)


def iterfiles(basepath: Path):
    for path in basepath.rglob("*"):
        if path.suffix.lower() in (".jpeg", ".jpg", ".heic"):
            yield path


def gps_dms_to_dd(dms: Sequence[Fraction]) -> float:
    """Degrees Minutes Seconds to Decimal Degrees"""

    return float(dms[0] + dms[1] / 60 + dms[2] / 3600)


def piexif_get_gps_value(d: dict[str, dict[int, Any]], idx2: int) -> Any:
    try:
        val = d["GPS"][idx2]
    except KeyError:
        return None

    try:
        return tuple(Fraction(*v) for v in val)
    except (ValueError, TypeError) as e:
        logging.warning("Invalid exif value. %s: %s", type(e).__name__, e)
        return None
    except ZeroDivisionError:
        return None


def get_exif(path: Path) -> tuple[float, float] | None:
    try:
        img = Image.open(path)
    except OSError as e:
        logging.error("Failed to open %s: %s", path, e)
        return None

    with img:
        if "exif" not in img.info:
            return None

        try:
            exif = piexif.load(img.info["exif"])
        except (ValueError, TypeError) as e:
            logging.warning("Failed to parse exif for %s: %s %s", path, type(e).__name__, e)
            return None
        except struct.error as e:
            logging.warning("Failed to parse exif for %s: %s", path, e)
            return None

        _lat = piexif_get_gps_value(exif, piexif.GPSIFD.GPSLatitude)
        _lon = piexif_get_gps_value(exif, piexif.GPSIFD.GPSLongitude)
        if _lat is None or _lon is None:
            return None

        lat = gps_dms_to_dd(_lat)
        lon = gps_dms_to_dd(_lon)
    return lat, lon


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--lat-lon",
        nargs=2,
        metavar=("LAT", "LON"),
        type=float,
        help="Resolve latitude and longitude in decimal degree format to Taiwan village",
    )
    group.add_argument("--path", type=Path, help="Analysis villages from picture directory")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("taiwan-village-photo-counts.csv"),
        help="Output path for the village statistics",
    )
    args = parser.parse_args()

    handler = RichHandler(log_time_format="%Y-%m-%d %H-%M-%S%Z", highlighter=NullHighlighter())
    FORMAT = "%(message)s"
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, handlers=[handler])
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=[handler])

    test_loopup()

    geoloc = TaiwanGeoLocation()

    if args.path:
        if args.out_csv.exists():
            parser.error(f"CSV file {args.out_csv.resolve()} already exists")

        files_with_gps = 0
        files_without_gps = 0
        files_not_in_taiwan = 0
        villages_d: DefaultDict[str, int] = defaultdict(int)
        invalid_files = 0

        columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )

        with RichProgress(*columns) as p:
            progress = Progress(p)
            for path in progress.track(iterfiles(args.path)):
                try:
                    try:
                        latlon = get_exif(path)
                    except UnidentifiedImageError as e:
                        logging.warning("Failed to load %s: %s", path, e)
                        invalid_files += 1
                        continue

                    if latlon is None:
                        files_without_gps += 1
                        continue

                    files_with_gps += 1
                    lat, lon = latlon
                    try:
                        info = geoloc.lookup(lat, lon)
                        s = f"{info['level1_chinese']}, {info['level2_chinese']}, {info['level3_chinese']}"
                        villages_d[s] += 1
                    except KeyError:
                        files_not_in_taiwan += 1
                except Exception:
                    logging.exception("Unhandled exception when loading %s", path)

        print("Number of files with GPS info", files_with_gps)
        print("Number of files without GPS info", files_without_gps)
        print("Number of files not within Taiwan", files_not_in_taiwan)
        print("Number of invalid files", invalid_files)

        villages = sorted(villages_d.items(), key=itemgetter(1))
        print("Number of villages", len(villages))

        with args.out_csv.open("x", newline="", encoding="utf-8") as fw:
            writer = csv.writer(fw)
            writer.writerow(["village", "count"])
            writer.writerows(villages)

        print("Written CSV village file to", args.out_csv.resolve())

    else:
        lat, lon = args.lat_lon
        info = geoloc.lookup(lat, lon)
        print(info)


if __name__ == "__main__":
    main()
