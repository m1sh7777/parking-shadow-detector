# re-downloads the riyadh parking spots from openstreetmap (via the HDX
# extract) and saves them to riyadh_parking.geojson. no api key needed.
#
#   python riyadh/fetch_parking.py
import json
import os
import urllib.request
import zipfile

HDX_URL = ("https://production-raw-data-api.s3.amazonaws.com/ISO3/SAU/"
           "points_of_interest/hotosm_sau_points_of_interest_osm_geojson.zip")
BBOX = (46.50, 24.55, 47.00, 24.95)   # riyadh city, lon/lat bounds
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "_sau_poi.zip")
OUT = os.path.join(HERE, "riyadh_parking.geojson")


def main():
    # skip the download if the zip is already here
    if not os.path.exists(TMP) or os.path.getsize(TMP) < 1_000_000:
        print(f"downloading {HDX_URL}")
        urllib.request.urlretrieve(HDX_URL, TMP)
    with zipfile.ZipFile(TMP) as z:
        with z.open("points_of_interest.geojson") as f:
            data = json.load(f)

    # keep only parking spots inside the riyadh box
    park = []
    for ft in data.get("features", []):
        props = ft.get("properties") or {}
        geom = ft.get("geometry") or {}
        if props.get("amenity") != "parking" or geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        if BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]:
            park.append({
                "name": props.get("name") or "unnamed",
                "lat": round(lat, 6), "lon": round(lon, 6),
                "access": props.get("access", "yes"),
                "fee": props.get("fee", ""),
                "capacity": props.get("capacity", ""),
                "parking": props.get("parking", ""),
            })

    fc = {
        "type": "FeatureCollection",
        "metadata": {
            "source": "OpenStreetMap contributors, via HDX hotosm_sau_points_of_interest",
            "bbox_filter": "24.55-24.95N, 46.50-47.00E (Riyadh city)",
            "amenity": "parking", "count": len(park),
        },
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
             "properties": {k: v for k, v in p.items() if k not in ("lat", "lon")}}
            for p in park
        ],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    os.remove(TMP)
    print(f"saved {OUT} ({len(park)} parking facilities)")


if __name__ == "__main__":
    main()
