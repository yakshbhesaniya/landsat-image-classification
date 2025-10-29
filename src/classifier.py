# classifier.py
# Manual K-means implementation for image classification into 3 classes (water, vegetation, other).
# No sklearn, no ready-made clustering functions.

import numpy as np

def _init_centroids(X, k, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    return X[idx].astype(np.float64)

def _assign_labels(X, centroids):
    # X: (n, d), centroids: (k, d)
    # returns labels (n,)
    # compute squared distances efficiently
    dists = np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=2)
    labels = np.argmin(dists, axis=1)
    return labels

def _compute_centroids(X, labels, k):
    d = X.shape[1]
    centroids = np.zeros((k, d), dtype=np.float64)
    for ki in range(k):
        members = X[labels == ki]
        if len(members) == 0:
            # reinitialize with a random data point
            centroids[ki] = X[np.random.randint(0, X.shape[0])]
        else:
            centroids[ki] = members.mean(axis=0)
    return centroids

def kmeans(X, k=3, max_iter=100, tol=1e-4, seed=0):
    """
    Basic K-means implementation.
    X: (n, d) float array
    Returns: labels (n,), centroids (k,d)
    """
    centroids = _init_centroids(X, k, seed)
    for it in range(max_iter):
        labels = _assign_labels(X, centroids)
        new_centroids = _compute_centroids(X, labels, k)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break
    return labels, centroids

def kmeans_classify(image_arr, k=3, max_iter=100, tol=1e-4, random_seed=0):
    """
    image_arr: (H, W, 3) float image values normalized (e.g., 0..1)
               order expected: [G, R, NIR] in the 3 channels (this GUI uses that order)
    returns classified_map (H, W) where:
        1 => water
        2 => vegetation
        3 => other
    """
    H, W, C = image_arr.shape
    assert C >= 3, "image_arr must have at least 3 channels (G,R,NIR)"
    # Build feature vector for clustering.
    # We'll use [NDVI, NDWI, brightness] where:
    # NDVI = (NIR - Red) / (NIR + Red)
    # NDWI = (Green - NIR) / (Green + NIR)
    green = image_arr[:,:,0].astype(np.float64)
    red = image_arr[:,:,1].astype(np.float64)
    nir = image_arr[:,:,2].astype(np.float64)
    eps = 1e-8
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    brightness = (red + green + nir) / 3.0

    # Stack features and flatten
    feats = np.stack([ndvi, ndwi, brightness], axis=-1)  # (H, W, 3)
    X = feats.reshape(-1, 3)

    # Run kmeans
    labels, centroids = kmeans(X, k=k, max_iter=max_iter, tol=tol, seed=random_seed)

    # Map clusters to classes. We'll use centroid feature means to decide:
    # - water: cluster with highest NDWI (water tends to have positive NDWI and low NIR)
    # - vegetation: cluster with highest NDVI
    # - other: remaining cluster
    cent_ndvi = centroids[:,0]
    cent_ndwi = centroids[:,1]
    # find indices
    water_idx = int(np.argmax(cent_ndwi))           # highest NDWI -> water
    vegetation_idx = int(np.argmax(cent_ndvi))      # highest NDVI -> vegetation
    # other is the one not water or vegetation
    all_idx = set(range(k))
    other_idx = (all_idx - {water_idx, vegetation_idx})
    if len(other_idx) == 1:
        other_idx = int(other_idx.pop())
    else:
        # possible conflict if both heuristics picked same cluster (rare) -> pick remaining by brightness
        remaining = list(all_idx - {water_idx})
        other_idx = remaining[0] if len(remaining) == 1 else remaining[0]

    # build mapping from cluster index -> class label
    mapping = {}
    mapping[water_idx] = 1
    mapping[vegetation_idx] = 2
    mapping[other_idx] = 3

    # Remap labels to class values
    mapped = np.vectorize(lambda lab: mapping.get(int(lab), 3))(labels)
    classified = mapped.reshape(H, W).astype(np.int32)
    return classified
