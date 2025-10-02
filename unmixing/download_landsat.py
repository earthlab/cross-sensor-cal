import os
import boto3
import geopandas as gpd
import pandas as pd
import botocore

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ECO_SHP = "data/Ecoregion/us_eco_l3.shp"
WRS2_SHP = "data/WRS2_descending_0/wrs2_descending.shp"
TARGET_REGION = "Middle Rockies"

START_DATE = "20230703"  # YYYYMMDD
END_DATE = "20230719"  # YYYYMMDD

OUT_DIR = "landsat8_nbar_downloads"
BUCKET = "usgs-landsat"  # USGS requester-pays bucket
PRODUCT = "collection02/level-2/standard/oli-tirs"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_tiles_for_region(ecoshp, wrsshp, region_name):
    eco = gpd.read_file(ecoshp)
    target = eco.loc[eco["US_L3NAME"] == region_name, "geometry"].unary_union
    wrs = gpd.read_file(wrsshp).to_crs(eco.crs)
    overlap = wrs[wrs.intersects(target)]
    # Return each as zero-padded strings
    return [(int(r.PATH), int(r.ROW)) for _, r in overlap.iterrows()]


def daterange(start_ymd, end_ymd):
    for dt in pd.date_range(
        start=pd.to_datetime(start_ymd, format="%Y%m%d"),
        end=pd.to_datetime(end_ymd, format="%Y%m%d"),
        freq="D",
    ):
        yield dt.strftime("%Y%m%d")


def download_nbar_for_tile_date(s3, bucket, product, path, row, yyyymmdd, out_base):
    year = yyyymmdd[:4]
    base_prefix = f"{product}/{year}/{path:03d}/{row:03d}/"
    paginator = s3.get_paginator("list_objects_v2")

    # List scene folders
    pages = paginator.paginate(
        Bucket=bucket, Prefix=base_prefix, Delimiter="/", RequestPayer="requester"
    )

    scene_prefixes = []
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            prefix = cp["Prefix"]  # e.g. .../LC08_L2SP_028032_20230703_20230714_02_T1/
            scene_id = prefix.rstrip("/").split("/")[-1]
            acqdate = scene_id.split("_")[3]
            if acqdate == yyyymmdd:
                scene_prefixes.append(prefix)

    # Download only the SR_B1–SR_B7 files and quality band for each matching scene
    for scene_prefix in scene_prefixes:
        scene_id = scene_prefix.rstrip("/").split("/")[-1]
        for band in range(1, 8):
            filename = f"{scene_id}_SR_B{band}.TIF"
            key = f"{scene_prefix}{filename}"
            local_dir = os.path.join(out_base, f"{path:03d}{row:03d}", yyyymmdd)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            if os.path.exists(local_path):
                print(f"Skipping {local_path}")
                continue
            print(f"Downloading s3://{bucket}/{key}")
            try:
                s3.download_file(
                    bucket, key, local_path, ExtraArgs={"RequestPayer": "requester"}
                )
            except Exception as e:
                print(f"  ↻ failed to download {key}: {e}")

        filename = f"{scene_id}_QA_PIXEL.TIF"
        key = f"{scene_prefix}{filename}"
        local_dir = os.path.join(out_base, f"{path:03d}{row:03d}", yyyymmdd)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            print(f"Skipping {local_path}")
            continue
        print(f"Downloading s3://{bucket}/{key}")
        try:
            s3.download_file(
                bucket, key, local_path, ExtraArgs={"RequestPayer": "requester"}
            )
        except Exception as e:
            print(f"  ↻ failed to download {key}: {e}")


def _no_iam_auth_client():
    try:
        s3_resource = boto3.resource(
            "s3",
            aws_access_key_id="",
            aws_secret_access_key="",
            aws_session_token=(
                os.environ["AWS_SESSION_TOKEN"]
                if "AWS_SESSION_TOKEN" in os.environ
                else ""
            ),
        )
        return boto3.client(
            "s3",
            aws_access_key_id="",
            aws_secret_access_key="",
            aws_session_token=(
                os.environ["AWS_SESSION_TOKEN"]
                if "AWS_SESSION_TOKEN" in os.environ
                else ""
            ),
        )
    except botocore.exceptions.ClientError:
        raise ValueError(f"Invalid AWS credentials or bucket {BUCKET} does not exist.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1) Determine overlapping WRS-2 tiles
    tiles = get_tiles_for_region(ECO_SHP, WRS2_SHP, TARGET_REGION)
    print(f"Tiles overlapping “{TARGET_REGION}”: {tiles}")

    # 2) Create a signed S3 client (will pick up your AWS creds)
    s3 = _no_iam_auth_client()

    # 3) Loop over tiles and dates, download NBAR bands
    for path, row in tiles:
        for ymd in daterange(START_DATE, END_DATE):
            download_nbar_for_tile_date(s3, BUCKET, PRODUCT, path, row, ymd, OUT_DIR)


if __name__ == "__main__":
    main()
