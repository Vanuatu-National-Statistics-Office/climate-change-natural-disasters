import os
import math
from datetime import datetime, timedelta
import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import pyproj
from shapely.ops import transform
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from pystac_client import Client
import odc.stac
from tqdm import tqdm

ADMIN_GEOZIP = "./AC2022.zip"
OUTPUT_DIR = "soil_health_output"
YEAR = 2017 # Iterate range for each year of composite (2017-2024)
DATERANGE_START = f"{YEAR}-01-01"
DATERANGE_END = f"{YEAR}-12-31"
AWS_STAC_URL = "http://stac.digitalearthpacific.org/"
CRS_OUT = "EPSG:32759"   # target CRS
BANDS = ["red", "green", "blue", "nir", "swir16"]
RESOLUTION = 10
CHUNKS = {'x': 1024, 'y': 1024, 'bands': -1, 'time': -1}

os.makedirs(OUTPUT_DIR, exist_ok=True)

admin_boundaries_gdf = gpd.read_file(ADMIN_GEOZIP)
print(f"Loaded {len(admin_boundaries_gdf)} polygons from {ADMIN_GEOZIP}")

stac_client = Client.open(AWS_STAC_URL)

def compute_ndmi(nir, swir):
    return (nir - swir) / (nir + swir)

def compute_bsi(blue, red, nir, swir):
    return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))

def compute_ndvi(red, nir):
    return (nir - red) / (nir + red)

def compute_savi(red, nir, L=0.5):
    return ((nir - red) * (1 + L)) / (nir + red + L)

def compute_msavi2(red, nir):
    a = 2 * nir + 1
    b = 2 * (nir**2 - nir * red)
    return (a - np.sqrt(a**2 - b)) / 2

def compute_nbr(nir, swir16):
    return (nir - swir16) / (nir + swir16)


results = []

for idx, row in tqdm(admin_boundaries_gdf.reset_index().iterrows(), total=len(admin_boundaries_gdf), desc="Processing polygons"):
    geom = row.geometry
    raw_name = str(row.get("ACNAME22", f"admin_{idx}"))
    safe_name = raw_name.replace(" ", "_").replace("-", "_")
    print(f"\nProcessing {raw_name} ...")

    # Convert to Lat/Lon degrees for STAC query
    geom_ll = geom
    if admin_boundaries_gdf.crs and admin_boundaries_gdf.crs.to_string() != "EPSG:4326":
        project_to_wgs = pyproj.Transformer.from_crs(admin_boundaries_gdf.crs, "EPSG:4326", always_xy=True).transform
        geom_ll = transform(project_to_wgs, geom)

    minx, miny, maxx, maxy = geom_ll.bounds
    bbox = (minx, miny, maxx, maxy)

    # STAC search for Sentinel-2 GeoMAD items
    s2_search = stac_client.search(
        collections=["dep_s2_geomad"],
        bbox=bbox,
        datetime=f"{DATERANGE_START}/{DATERANGE_END}",
    )
    s2_items = s2_search.item_collection()

    if len(s2_items) == 0:
        print(f"  -> No Sentinel-2 items found for {raw_name}, skipping.")
        # Add empty index columns for consistency
        results.append({**row, **{f"{i}_mean": np.nan for i in ['ndmi','bsi','ndvi','savi','msavi2','nbr']}})
        continue

    try:
        s2_data = odc.stac.load(
            items=s2_items,
            bands=BANDS,
            bbox=bbox,
            crs=CRS_OUT,
            chunks=CHUNKS,
            resolution=RESOLUTION
        )
    except Exception as e:
        print(f"  -> odc.stac.load failed for {raw_name}: {e}")
        results.append({**row, **{f"{i}_mean": np.nan for i in ['ndmi','bsi','ndvi','savi','msavi2','nbr']}})
        continue

    # Project geometry to match imagery CRS
    if admin_boundaries_gdf.crs and admin_boundaries_gdf.crs.to_string() != CRS_OUT:
        project_geom = pyproj.Transformer.from_crs(admin_boundaries_gdf.crs, CRS_OUT, always_xy=True).transform
        geom_proj = transform(project_geom, geom)
    else:
        geom_proj = geom

    # Create mask for polygon
    mask = geometry_mask(
        geometries=[mapping(geom_proj)],
        transform=s2_data.odc.transform,
        out_shape=(s2_data.sizes["y"], s2_data.sizes["x"]),
        invert=True
    )
    mask_xr = xr.DataArray(mask, dims=("y", "x"), coords={"y": s2_data.y, "x": s2_data.x})

    s2_masked = s2_data.where(mask_xr)

    # Compute indices
    s2_masked = s2_masked.assign(
        ndmi=compute_ndmi(s2_masked['nir'], s2_masked['swir16']),
        bsi=compute_bsi(s2_masked['blue'], s2_masked['red'], s2_masked['nir'], s2_masked['swir16']),
        ndvi=compute_ndvi(s2_masked['red'], s2_masked['nir']),
        savi=compute_savi(s2_masked['red'], s2_masked['nir']),
        msavi2=compute_msavi2(s2_masked['red'], s2_masked['nir']),
        nbr=compute_nbr(s2_masked['nir'], s2_masked['swir16'])
    )

    # Compute mean of each index
    index_means = {}
    for index_name in ["ndmi", "bsi", "ndvi", "savi", "msavi2", "nbr"]:
        try:
            mean_val = (
                s2_masked[index_name]
                .mean(skipna=True)
                .compute()
                .item()
            )
        except Exception:
            mean_val = np.nan
        index_means[f"{index_name}_mean"] = mean_val

    print(f"  -> Index means: {index_means}")

    # Combine with original row data
    result_row = {**row.to_dict(), **index_means}
    results.append(result_row)


# Build output geodataframe
output_gdf = gpd.GeoDataFrame(results, crs=admin_boundaries_gdf.crs)

# Save as GeoJSON
output_geojson = os.path.join(OUTPUT_DIR, f"admin_index_means_{YEAR}.geojson")
output_gdf.to_file(output_geojson, driver="GeoJSON")
print(f"\nSaved final GeoJSON with index means to: {output_geojson}")
