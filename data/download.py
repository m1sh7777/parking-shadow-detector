# downloads the parking datasets used for testing.
# the old cnrpark.it links are dead, the files live on a github release now
# (config.MIRROR). downloads into data/raw/ and unzips automatically.
#
# usage:
#   python data/download.py            # full CNR-EXT (449.5 MB) + splits
#   python data/download.py --small    # just CNRPark (36.6 MB), quick test
import os
import sys
import zipfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _expected_size(url):
    # asks the server how big the file should be, so we can spot broken
    # partial downloads
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def fetch(url, dest):
    # download url to dest. skips it if the file is already fully there
    expected = _expected_size(url)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if expected and size == expected:
            print(f"already exists: {dest} ({size / 1e6:.1f} MB)")
            return dest
        print(f"incomplete ({size / 1e6:.1f} of {expected / 1e6:.1f} MB) - re-downloading")
        os.remove(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception:
        # dont leave a half downloaded file behind
        if os.path.exists(dest):
            os.remove(dest)
        raise
    print()
    return dest


def extract(zip_path, folder_name):
    # unzip into data/raw. some zips have the folder inside already,
    # some dont, this handles both
    out_dir = os.path.join(config.DATA_DIR, folder_name)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"already extracted: {out_dir}")
        return
    print(f"extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        has_root = any(n == folder_name + "/" or n.startswith(folder_name + "/")
                       for n in z.namelist())
        z.extractall(config.DATA_DIR if has_root else out_dir)
    print(f"extracted -> {out_dir if not has_root else config.DATA_DIR}")


def _progress(blocks, bs, total):
    if total > 0:
        pct = min(100.0, 100.0 * blocks * bs / total)
        sys.stdout.write(f"\r  {pct:5.1f}% of {total / 1e6:.1f} MB")
        sys.stdout.flush()


def main():
    small = "--small" in sys.argv
    if small:
        z = fetch(config.CNR_URL, os.path.join(config.DATA_DIR, "CNRPark-Patches-150x150.zip"))
        extract(z, "CNRPark-Patches-150x150")
    else:
        z = fetch(config.CNR_EXT_URL, os.path.join(config.DATA_DIR, "CNR-EXT-Patches-150x150.zip"))
        extract(z, "CNR-EXT-Patches-150x150")

    # splits.zip is optional, only needed for the official label file
    try:
        z = fetch(config.SPLITS_URL, os.path.join(config.DATA_DIR, "splits.zip"))
        extract(z, "splits")
    except Exception as e:
        print(f"[warn] splits.zip skipped ({e.__class__.__name__}: {e})")
        print("       Not needed for evaluate.py - only for official train/test splits.")

    print("\nDone. Next step:")
    if small:
        print("  python evaluate.py --data data/raw/CNRPark-Patches-150x150")
    else:
        print("  python evaluate.py --data data/raw/CNR-EXT-Patches-150x150 --limit 300")


if __name__ == "__main__":
    main()
