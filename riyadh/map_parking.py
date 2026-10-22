# draws the riyadh parking spots on a map, saves riyadh_parking_map.png
#
#   python riyadh/map_parking.py
#
# the street map under the dots is optional. if u pip install contextily
# u get the street tiles, without it u just get the dots on a plain bg
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "riyadh_parking.geojson")
OUT = os.path.join(HERE, "riyadh_parking_map.png")


def to_mercator(lon, lat):
    # normal lat/lon -> map projection coords so the tiles line up
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * 20037508.34 / 180
    return x, y


def main():
    fc = json.load(open(SRC, encoding="utf-8"))
    pts = [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1])
           for f in fc["features"]]
    xy = [to_mercator(lon, lat) for lon, lat in pts]

    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    ax.scatter([p[0] for p in xy], [p[1] for p in xy], s=26, c="#e63946",
               edgecolors="white", linewidths=0.6, zorder=5, alpha=0.9)

    # label the two obvious clusters so people know what theyre looking at
    labeled = {"King Khalid Intl. Airport": (46.599, 24.958),
               "Downtown / Al Malaz cluster": (46.72, 24.67)}
    for label, (lon, lat) in labeled.items():
        px, py = to_mercator(lon, lat)
        ax.annotate(label, (px, py), textcoords="offset points", xytext=(10, 8),
                    fontsize=9, fontweight="bold", color="#1d3557",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1d3557", alpha=0.85))

    try:
        import contextily as cx
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.Esri.WorldStreetMap,
                       zoom=11, attribution=False)
        tiles = "Tiles: (c) Esri, Maxar, Earthstar Geographics."
    except ImportError:
        ax.set_facecolor("#f2f2f0")
        tiles = "Basemap off (pip install contextily for street tiles)."

    ax.set_axis_off()
    ax.set_title(f"Parking facilities mapped in OpenStreetMap - Riyadh city ({len(pts)})",
                 fontsize=13, fontweight="bold", pad=10)
    fig.text(0.01, 0.01, "Data: (c) OpenStreetMap contributors via HDX. " + tiles,
             fontsize=7.5, color="#333333",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
