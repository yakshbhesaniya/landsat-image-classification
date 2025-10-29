# utils.py
# Image reading, band stacking, normalization utilities.
# Preferred reader: rasterio. Falls back to Pillow for simple multi-band TIFFs.

import numpy as np
import rasterio

def _normalize_uint_to_float(arr):
    """
    Convert numeric array to float in range 0..1 based on dtype max.
    """
    if arr.dtype.kind == 'f':
        # assume already float in 0..1 or reflectance; clip
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    # integer types
    iinfo = None
    try:
        import numpy as np
        iinfo = np.iinfo(arr.dtype)
        maxv = iinfo.max
    except Exception:
        maxv = arr.max() if arr.max() > 0 else 1.0
    arrf = arr.astype(np.float32) / float(maxv)
    arrf = np.clip(arrf, 0.0, 1.0)
    return arrf

#def read_image_bands(path, band_tuple):
    """
    Reads specified bands from a multi-band image file.
    path: path to image file (GeoTIFF or multi-band TIFF)
    band_tuple: (G_index, R_index, NIR_index) with 1-based indices as user inputs
    Returns:
      arr: numpy array shape (H, W, 3) with channel order (G, R, NIR) as float32 normalized 0..1
      meta: metadata dict (if available from rasterio) or None
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            count = src.count
            H = src.height
            W = src.width
            # Ensure requested band indices are within range
            g_idx, r_idx, nir_idx = band_tuple
            if g_idx > count or r_idx > count or nir_idx > count:
                raise ValueError(f"Band indices exceed band count ({count}) in file {path}")
            # read(1) is 1-based band reading in rasterio
            g = src.read(g_idx)
            r = src.read(r_idx)
            nir = src.read(nir_idx)
            # normalize each
            g = _normalize_uint_to_float(g)
            r = _normalize_uint_to_float(r)
            nir = _normalize_uint_to_float(nir)
            arr = np.stack([g, r, nir], axis=-1)
            meta = src.meta.copy()
            return arr, meta
    except Exception as e:
        # fallback to PIL for simple multi-band TIFFs
        try:
            from PIL import Image
            im = Image.open(path)
            # PIL band indices are 0-based; convert and attempt to access frames or bands
            # Convert to numpy: attempt to read as multi-band
            arr_all = np.array(im)
            # arr_all may be (H,W,bands) or (H,W) for single band
            if arr_all.ndim == 2:
                raise RuntimeError("Single-band image; cannot extract 3 bands.")
            # assume arr_all has bands in last axis
            g_idx, r_idx, nir_idx = [i-1 for i in band_tuple]
            arr_g = arr_all[..., g_idx]
            arr_r = arr_all[..., r_idx]
            arr_nir = arr_all[..., nir_idx]
            arr_g = _normalize_uint_to_float(arr_g)
            arr_r = _normalize_uint_to_float(arr_r)
            arr_nir = _normalize_uint_to_float(arr_nir)
            arr = np.stack([arr_g, arr_r, arr_nir], axis=-1)
            return arr, None
        except Exception as e2:
            # can't read
            print("Error reading image (rasterio and PIL fallback failed):", e, e2)
            return None, None

def compute_ndvi_ndwi(red, nir, green):
    """
    Compute NDVI and NDWI from arrays.
    Inputs: arrays same shape (H,W)
    Returns: ndvi, ndwi arrays
    """
    eps = 1e-8
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    return ndvi, ndwi

def read_image_bands(red_path, green_path, nir_path):
    """Reads and stacks individual single-band TIFF files into one 3D array (H, W, 3)."""
    try:
        with rasterio.open(green_path) as g_src:
            green = g_src.read(1).astype(np.float32)
            meta = g_src.meta.copy()

        with rasterio.open(red_path) as r_src:
            red = r_src.read(1).astype(np.float32)

        with rasterio.open(nir_path) as n_src:
            nir = n_src.read(1).astype(np.float32)

        # Normalize to 0–1 range (optional but helps classification)
        def normalize(arr):
            arr = np.clip(arr, 0, np.percentile(arr, 99))
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

        red = normalize(red)
        green = normalize(green)
        nir = normalize(nir)

        # Stack into (H, W, 3): order = (G, R, NIR)
        stacked = np.dstack([green, red, nir])
        return stacked, meta

    except Exception as e:
        print("Error reading bands:", e)
        return None, None
