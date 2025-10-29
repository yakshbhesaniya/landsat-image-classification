# Landsat Image Classification — Powai Region (2005–2025)

This project performs **unsupervised land cover classification** on three **Landsat satellite images** of the Powai area (2005–2025).  
Each image is classified into **three classes** — *Water*, *Vegetation*, and *Others* — using a **manual K-Means algorithm** (no sklearn used).  
Results are displayed side-by-side in a **Tkinter GUI**, along with a comparison of water area across the years.

---

## Features
- GUI for selecting Red, Green, and NIR bands for 3 different images  
- Custom K-Means implementation (no external ML libraries)  
- Displays all 3 classified images simultaneously  
- Calculates and compares water area for each image  
- No files are saved — results are shown directly in the GUI

---

## Principle
Landsat multispectral images contain several spectral bands.  
By stacking the Red, Green, and NIR bands, each pixel is represented as a 3D feature vector.  
A manual **K-Means clustering** algorithm groups these pixels into three clusters based on spectral similarity.  
Clusters are labeled as:
- **Lowest NIR reflectance** → Water  
- **Highest NIR reflectance** → Vegetation  
- Remaining → Others

---

## How to Run
cd src
python main.py

## Output
- Three classified maps (for 3 years) displayed side-by-side
- Water area for each image printed and shown in the GUI