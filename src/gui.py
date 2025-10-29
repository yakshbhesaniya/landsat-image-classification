# Tkinter GUI for selecting R, G, NIR bands for 3 images and showing results.
# Displays 3 classified images side-by-side and a water-area comparison bar chart.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.utils import read_image_bands
from src.classifier import kmeans_classify
from src.area_calculator import compute_water_area_km2

class LandsatClassifierGUI:
    def __init__(self, root_title="Landsat 3-Image Classifier (KMeans, k=3)"):
        self.root = tk.Tk()
        self.root.title(root_title)
        self.root.geometry("1200x750")

        # Store band filepaths per image
        self.band_files = [
            {"red": tk.StringVar(), "green": tk.StringVar(), "nir": tk.StringVar()},
            {"red": tk.StringVar(), "green": tk.StringVar(), "nir": tk.StringVar()},
            {"red": tk.StringVar(), "green": tk.StringVar(), "nir": tk.StringVar()}
        ]
        self.pixel_res_var = tk.StringVar(value="30")

        self._build_widgets()

    def _build_widgets(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(frm)
        top.pack(fill=tk.X)

        # For 3 images — each with 3 band selectors
        for i in range(3):
            sub = ttk.Labelframe(top, text=f"Image {i+1}", padding=8)
            sub.grid(row=0, column=i, padx=8, pady=6, sticky="nsew")

            for j, band in enumerate(["red", "green", "nir"]):
                ttk.Label(sub, text=f"{band.upper()} Band:").grid(row=j, column=0, sticky="e", pady=4)
                ent = ttk.Entry(sub, textvariable=self.band_files[i][band], width=35)
                ent.grid(row=j, column=1, padx=4)
                ttk.Button(sub, text="Browse", command=lambda idx=i, b=band: self._browse_band(idx, b)).grid(row=j, column=2, padx=4)

        # Pixel resolution input and Run button
        controls = ttk.Frame(frm)
        controls.pack(fill=tk.X, pady=6)

        ttk.Label(controls, text="Pixel resolution (m):").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(controls, textvariable=self.pixel_res_var, width=6).grid(row=0, column=1, sticky="w")
        ttk.Button(controls, text="Run Classification (K-Means, k=3)", command=self._on_run).grid(row=0, column=2, padx=20)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status).grid(row=0, column=3, padx=10)

        # Figure area for classified maps + water area comparison
        display_frame = ttk.Frame(frm)
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11, 6))
        self.axs = [self.fig.add_subplot(2, 2, i + 1) for i in range(4)]
        self._clear_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _clear_axes(self):
        for ax in getattr(self, "axs", []):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

    def _browse_band(self, idx, band):
        path = filedialog.askopenfilename(
            title=f"Select {band.upper()} band for Image {idx+1}",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if path:
            self.band_files[idx][band].set(path)

    def _on_run(self):
        # Validate inputs
        for i, bands in enumerate(self.band_files):
            for bname, var in bands.items():
                if not var.get().strip():
                    messagebox.showerror("Missing band", f"Please select {bname.upper()} band for Image {i+1}")
                    return

        try:
            pixel_res = float(self.pixel_res_var.get())
            if pixel_res <= 0:
                raise ValueError()
        except Exception:
            messagebox.showerror("Invalid pixel size", "Pixel resolution must be a positive number.")
            return

        # Disable UI during processing
        self.status.set("Running...")
        self.root.config(cursor="watch")

        threading.Thread(target=self._run_pipeline, args=(pixel_res,), daemon=True).start()

    def _run_pipeline(self, pixel_res):
        try:
            classified_maps = []
            water_areas = []
            years = []

            for i, bands in enumerate(self.band_files):
                red_path = bands["red"].get()
                green_path = bands["green"].get()
                nir_path = bands["nir"].get()

                # Read and stack RGB (actually G,R,NIR) to 3D array normalized [0,1]
                arr, meta = read_image_bands(red_path, green_path, nir_path)
                if arr is None:
                    raise RuntimeError(f"Error reading image {i+1} bands.")

                classified = kmeans_classify(arr, max_iter=80, tol=1e-4, random_seed=42)
                classified_maps.append(classified)

                water_km2 = compute_water_area_km2(classified, pixel_res)
                water_areas.append(water_km2)

                import re
                m = re.search(r"(19|20)\d{2}", red_path)
                years.append(m.group(0) if m else f"Image {i+1}")

            self._plot_results(classified_maps, water_areas, years)
            self.status.set("Done")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", str(e))
            self.status.set("Error")
        finally:
            self.root.config(cursor="")

    def _plot_results(self, classified_maps, water_areas, years):
        for ax in self.axs:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        cmap = {1: (0, 0, 1), 2: (0, 0.6, 0), 3: (0.7, 0.7, 0.7)}

        for i in range(3):
            ax = self.axs[i]
            cm = classified_maps[i]
            h, w = cm.shape
            rgb = np.zeros((h, w, 3), dtype=float)
            for cls_val, col in cmap.items():
                mask = (cm == cls_val)
                for ch in range(3):
                    rgb[:, :, ch][mask] = col[ch]
            ax.imshow(rgb)
            ax.set_title(f"{years[i]} - Water: {water_areas[i]:.4f} sq.km")

        axb = self.axs[3]
        axb.bar(years, water_areas, color="skyblue")
        axb.set_title("Water Area Comparison (sq.km)")
        axb.set_ylabel("Area (sq.km)")

        self.canvas.draw()

    def run(self):
        self.root.mainloop()

