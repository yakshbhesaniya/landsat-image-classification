# SIP Project - Image Classification - Complete Documentation

This document provides a comprehensive line-by-line explanation of the entire project, including all files, functions, and library usage.

---

## File-by-File Breakdown

### 1. src/main.py - Application Entry Point

**Libraries Used:** None (only imports from local modules)

#### Functions:

##### `main()` (lines 5-7)
- **Purpose:** Entry point function that initializes and runs the GUI application
- **Line 5:** Defines the main function
- **Line 6:** Creates an instance of `LandsatClassifierGUI` class and stores it in variable `app`
- **Line 7:** Calls the `run()` method on the app instance to start the GUI event loop

##### `if __name__ == "__main__":` (lines 9-10)
- **Purpose:** Python idiom to ensure code only runs when script is executed directly (not when imported)
- **Line 9:** Checks if the script is being run as the main program
- **Line 10:** If true, calls the `main()` function to start the application

---

### 2. src/gui.py - GUI Application Class

**Libraries Used:** 
- `tkinter` (tk): Python's standard GUI library for creating windows and widgets
- `tkinter.ttk`: Themed widgets (modern-looking UI components)
- `tkinter.filedialog`: File selection dialogs
- `tkinter.messagebox`: Error and information popup dialogs
- `threading`: Allows background processing without freezing the GUI
- `numpy` (np): Numerical computing library for array operations
- `matplotlib`: Plotting and visualization library
- `matplotlib.pyplot` (plt): High-level plotting interface
- `matplotlib.backends.backend_tkagg.FigureCanvasTkAgg`: Embeds matplotlib plots in Tkinter

#### Class: LandsatClassifierGUI

**Instance Variables:**
- `self.root`: Main Tkinter window object
- `self.band_files`: List of 3 dictionaries, each storing file paths for red, green, and nir bands
- `self.pixel_res_var`: Tkinter StringVar storing pixel resolution value
- `self.status`: Tkinter StringVar for displaying application status
- `self.fig`: Matplotlib Figure object for plotting
- `self.axs`: List of 4 matplotlib Axes objects (3 for images, 1 for bar chart)
- `self.canvas`: Matplotlib canvas widget embedded in Tkinter

#### Functions:

##### `__init__(self, root_title="Landsat 3-Image Classifier (KMeans, k=3)")` (lines 17-30)
- **Purpose:** Constructor that initializes the GUI application
- **Line 17:** Defines the initialization method with optional window title parameter
- **Line 18:** Creates the main Tkinter root window object
- **Line 19:** Sets the window title using the provided parameter
- **Line 20:** Sets the initial window size to 1200x750 pixels
- **Lines 23-27:** Creates a list containing 3 dictionaries, each with StringVar objects for storing file paths of red, green, and nir bands for each image
- **Line 28:** Creates a StringVar initialized with default value "30" for pixel resolution
- **Line 30:** Calls the `_build_widgets()` method to construct the user interface

##### `_build_widgets(self)` (lines 32-77)
- **Purpose:** Constructs all GUI widgets and layouts the interface
- **Line 33:** Creates a main frame with 8 pixels padding around edges
- **Line 34:** Packs the main frame to fill and expand in both directions
- **Line 36:** Creates a top frame for image input sections
- **Line 37:** Packs the top frame to fill horizontally (X direction)
- **Lines 40-41:** Configures column weights for responsive layout - each of the 3 columns gets equal weight and uniform sizing
- **Lines 44-55:** Loop that creates 3 image input sections:
  - **Line 45:** Creates a labeled frame for each image (Image 1, Image 2, Image 3) with 8 pixels padding
  - **Line 46:** Places each labeled frame in a grid layout (row 0, columns 0, 1, 2) with padding and sticky expansion
  - **Line 49:** Configures the entry field column to expand when window resizes
  - **Lines 51-55:** Inner loop creating 3 band input rows (RED, GREEN, NIR) for each image:
    - **Line 52:** Creates a label for the band name, positioned on the right side
    - **Line 53:** Creates an entry field with width 20, linked to the corresponding StringVar
    - **Line 54:** Places the entry field in the grid, allowing it to expand horizontally
    - **Line 55:** Creates a Browse button that calls `_browse_band()` with the image index and band name
- **Line 58:** Creates a frame for control buttons (pixel resolution and run button)
- **Line 59:** Packs the controls frame horizontally with vertical padding
- **Line 61:** Creates a label for "Pixel resolution (m):"
- **Line 62:** Creates an entry field for pixel resolution input, width 6 characters
- **Line 63:** Creates the "Run Classification" button that triggers `_on_run()` method
- **Line 65:** Creates a StringVar for status display, initialized with "Ready"
- **Line 66:** Creates a label that displays the status StringVar value
- **Line 69:** Creates a frame for the display area (plots and charts)
- **Line 70:** Packs the display frame to fill and expand in both directions
- **Line 72:** Creates a matplotlib Figure with size 11x6 inches
- **Line 73:** Creates 4 subplots in a 2x2 grid layout and stores them in a list
- **Line 74:** Calls `_clear_axes()` to initialize empty plots
- **Line 76:** Creates a matplotlib canvas widget that embeds the figure in the Tkinter frame
- **Line 77:** Packs the canvas widget to fill and expand in the display frame

##### `_clear_axes(self)` (lines 79-83)
- **Purpose:** Clears all plot axes and removes tick marks
- **Line 80:** Loops through all axes in `self.axs` (if they exist)
- **Line 81:** Clears all plotted content from the axis
- **Line 82:** Removes x-axis tick marks
- **Line 83:** Removes y-axis tick marks

##### `_browse_band(self, idx, band)` (lines 85-91)
- **Purpose:** Opens a file dialog to select a band file for a specific image
- **Parameters:** `idx` - image index (0, 1, or 2), `band` - band name ("red", "green", or "nir")
- **Line 86:** Opens a file selection dialog with a title indicating which band and image
- **Line 88:** Sets file type filter to show only TIFF files (*.tif, *.tiff) or all files
- **Line 90:** Checks if a file path was selected (not cancelled)
- **Line 91:** Sets the selected file path to the corresponding StringVar in `self.band_files`

##### `_on_run(self)` (lines 93-113)
- **Purpose:** Validates inputs and starts the classification process in a background thread
- **Lines 95-99:** Validates that all band files are selected for all 3 images:
  - **Line 95:** Loops through each image's band dictionary
  - **Line 96:** Loops through each band (red, green, nir) in the dictionary
  - **Line 97:** Checks if the file path StringVar is empty or contains only whitespace
  - **Line 98:** Shows an error message if any band is missing
  - **Line 99:** Returns early if validation fails
- **Lines 101-107:** Validates pixel resolution input:
  - **Line 102:** Attempts to convert the pixel resolution StringVar to a float
  - **Line 103:** Checks if the value is positive (greater than 0)
  - **Line 104:** Raises a ValueError if validation fails
  - **Line 105:** Catches any exception during conversion or validation
  - **Line 106:** Shows an error message if pixel resolution is invalid
  - **Line 107:** Returns early if validation fails
- **Line 110:** Updates status StringVar to "Running..." to inform the user
- **Line 111:** Changes the cursor to a watch/hourglass icon to indicate processing
- **Line 113:** Starts a background thread running `_run_pipeline()` with pixel_res parameter, daemon=True ensures thread stops when main program exits

##### `_run_pipeline(self, pixel_res)` (lines 115-150)
- **Purpose:** Processes all 3 images, performs classification, and displays results
- **Parameter:** `pixel_res` - pixel resolution in meters (float)
- **Line 116:** Wraps code in try-except for error handling
- **Lines 117-119:** Initializes empty lists to store classification results, water areas, and year labels
- **Lines 121-139:** Loops through each of the 3 images:
  - **Lines 122-124:** Retrieves file paths for red, green, and nir bands from StringVars
  - **Line 127:** Calls `read_image_bands()` from utils module to read and stack the three band files into a 3D numpy array
  - **Line 128:** Checks if image reading failed (returns None)
  - **Line 129:** Raises a RuntimeError with descriptive message if reading failed
  - **Line 131:** Calls `kmeans_classify()` from classifier module to classify the image into 3 classes
  - **Line 132:** Appends the classified map to the results list
  - **Line 134:** Calls `compute_water_area_km2()` from area_calculator module to calculate water area in square kilometers
  - **Line 135:** Appends the calculated water area to the list
  - **Line 137:** Imports the `re` module for regular expression matching
  - **Line 138:** Searches the file path for a 4-digit year (starting with 19 or 20)
  - **Line 139:** Extracts the year if found, otherwise uses "Image {i+1}" as label
- **Line 141:** Calls `_plot_results()` to display all results in the GUI
- **Line 142:** Updates status to "Done" when processing completes successfully
- **Lines 144-148:** Error handling:
  - **Line 144:** Catches any exception during processing
  - **Line 145:** Imports traceback module for detailed error information
  - **Line 146:** Prints full error traceback to console for debugging
  - **Line 147:** Shows an error message dialog to the user
  - **Line 148:** Updates status to "Error"
- **Line 150:** Resets the cursor to default (removes watch icon) in the finally block

##### `_plot_results(self, classified_maps, water_areas, years)` (lines 152-218)
- **Purpose:** Displays classified images and bar chart with legend
- **Parameters:** 
  - `classified_maps`: List of 3 classified 2D arrays
  - `water_areas`: List of 3 water area values in sq.km
  - `years`: List of 3 year/image labels
- **Lines 154-158:** Clears only the first 3 axes (image displays), not the bar chart:
  - **Line 154:** Loops through first 3 axes (indices 0, 1, 2)
  - **Line 155:** Gets reference to each axis
  - **Line 156:** Clears all plotted content
  - **Line 157:** Removes x-axis ticks
  - **Line 158:** Removes y-axis ticks
- **Line 160:** Defines color mapping dictionary: 1=blue (water), 2=green (vegetation), 3=gray (others)
- **Line 161:** Defines label mapping dictionary for legend
- **Lines 163-173:** Loops through each of the 3 classified images:
  - **Line 164:** Gets reference to the corresponding axis
  - **Line 165:** Gets the classified map array for this image
  - **Line 166:** Gets height and width dimensions of the classified map
  - **Line 167:** Creates an empty RGB array (height x width x 3 channels) filled with zeros
  - **Lines 168-171:** Maps classification values to colors:
    - **Line 168:** Loops through each classification value and its color
    - **Line 169:** Creates a boolean mask for pixels with this classification value
    - **Line 170:** Loops through RGB channels (0, 1, 2)
    - **Line 171:** Sets the RGB channel value for masked pixels to the corresponding color component
  - **Line 172:** Displays the RGB image on the axis using `imshow()`
  - **Line 173:** Sets the title showing year/image label and water area value
- **Lines 176-196:** Creates and configures the bar chart (4th subplot):
  - **Line 176:** Gets reference to the 4th axis (bar chart)
  - **Line 177:** Clears any existing content
  - **Line 178:** Creates bar chart with years on x-axis, water_areas on y-axis, skyblue color with navy edges
  - **Line 179:** Sets chart title with bold font
  - **Line 180:** Sets y-axis label
  - **Line 181:** Sets x-axis label
  - **Lines 184-185:** Configures tick label font sizes for both axes
  - **Line 186:** Adds horizontal grid lines with transparency for readability
  - **Lines 189-193:** Adds value labels on top of each bar:
    - **Line 189:** Loops through each bar and its corresponding value
    - **Line 190:** Gets the height of the bar
    - **Lines 191-193:** Adds text label at the top center of each bar showing the exact value
  - **Line 196:** Rotates x-axis labels 45 degrees if any label is longer than 10 characters
- **Line 199:** Adjusts subplot layout to leave 20% space on the left side for the legend
- **Lines 202-216:** Creates and positions the classification legend:
  - **Line 202:** Imports Patch class from matplotlib.patches for creating legend elements
  - **Lines 203-204:** Creates legend elements (colored patches) for each classification class
  - **Line 206:** Gets reference to Image 1 axis (top-left, index 0)
  - **Line 207:** Gets reference to Image 3 axis (bottom-left, index 2)
  - **Line 208:** Gets the bounding box (position) of Image 1
  - **Line 209:** Gets the bounding box of Image 3
  - **Line 211:** Calculates the vertical center point between Image 1 and Image 3
  - **Lines 213-216:** Creates the legend positioned on the left side, vertically centered between the two left images
- **Line 218:** Redraws the canvas to display all updates

##### `run(self)` (lines 220-221)
- **Purpose:** Starts the Tkinter event loop to display and run the GUI
- **Line 221:** Calls `mainloop()` on the root window, which starts the GUI event loop and keeps the window open until closed

---

### 3. src/utils.py - Image Reading and Processing Utilities

**Libraries Used:**
- `numpy` (np): For array operations and numerical computations
- `rasterio`: Library for reading and writing geospatial raster data (GeoTIFF files)

#### Functions:

##### `_normalize_uint_to_float(arr)` (lines 7-24)
- **Purpose:** Converts numeric arrays to float values in the range 0.0 to 1.0
- **Parameter:** `arr` - numpy array (can be integer or float type)
- **Line 11:** Checks if the array data type is floating point
- **Line 13:** If float, clips values to 0.0-1.0 range and converts to float32
- **Lines 15-21:** For integer types:
  - **Line 15:** Initializes variable for integer info
  - **Lines 17-18:** Gets maximum value for the integer data type using numpy.iinfo()
  - **Line 19:** Extracts the maximum value
  - **Lines 20-21:** Handles exception if iinfo fails, uses array's actual max value or defaults to 1.0
- **Line 22:** Converts array to float32 and divides by maximum to normalize to 0-1 range
- **Line 23:** Clips the result to ensure values stay within 0.0-1.0 range
- **Line 24:** Returns the normalized float array

##### `compute_ndvi_ndwi(red, nir, green)` (lines 82-91)
- **Purpose:** Computes NDVI (Normalized Difference Vegetation Index) and NDWI (Normalized Difference Water Index) from spectral bands
- **Parameters:** 
  - `red`: Red band array (H, W)
  - `nir`: Near-Infrared band array (H, W)
  - `green`: Green band array (H, W)
- **Line 88:** Sets a small epsilon value (1e-8) to prevent division by zero
- **Line 89:** Calculates NDVI: (NIR - Red) / (NIR + Red + epsilon)
  - NDVI indicates vegetation health (higher values = more vegetation)
- **Line 90:** Calculates NDWI: (Green - NIR) / (Green + NIR + epsilon)
  - NDWI indicates water content (higher values = more water)
- **Line 91:** Returns both NDVI and NDWI arrays

##### `read_image_bands(red_path, green_path, nir_path)` (lines 93-121)
- **Purpose:** Reads three separate single-band TIFF files and stacks them into a 3D array
- **Parameters:**
  - `red_path`: File path to red band TIFF file
  - `green_path`: File path to green band TIFF file
  - `nir_path`: File path to near-infrared band TIFF file
- **Line 95:** Wraps code in try-except for error handling
- **Lines 96-98:** Opens and reads the green band file:
  - **Line 96:** Opens the green band file using rasterio context manager
  - **Line 97:** Reads the first band (index 1) and converts to float32
  - **Line 98:** Copies metadata (geospatial information, CRS, transform, etc.) from the green band file
- **Lines 100-101:** Opens and reads the red band file, converts to float32
- **Lines 103-104:** Opens and reads the NIR band file, converts to float32
- **Lines 107-109:** Defines a nested normalization function:
  - **Line 108:** Clips array values to 99th percentile to remove outliers
  - **Line 109:** Normalizes to 0-1 range using min-max scaling with small epsilon to prevent division by zero
- **Lines 111-113:** Applies normalization to each band (red, green, nir)
- **Line 116:** Stacks the three bands along the depth axis using `np.dstack()` to create (H, W, 3) array
  - Channel order: [Green, Red, NIR] as specified in the comment
- **Line 117:** Returns the stacked array and metadata
- **Lines 119-121:** Error handling:
  - **Line 119:** Catches any exception during file reading
  - **Line 120:** Prints error message to console
  - **Line 121:** Returns None, None to indicate failure

---

### 4. src/classifier.py - K-Means Classification Implementation

**Libraries Used:**
- `numpy` (np): For array operations, mathematical computations, and random number generation

#### Functions:

##### `_init_centroids(X, k, seed=0)` (lines 6-10)
- **Purpose:** Initializes K-means centroids by randomly selecting K data points
- **Parameters:**
  - `X`: Input data array of shape (n, d) where n is number of samples, d is number of features
  - `k`: Number of clusters (centroids) to initialize
  - `seed`: Random seed for reproducibility (default 0)
- **Line 7:** Creates a random number generator with the specified seed
- **Line 8:** Gets the number of data points (n) from the first dimension of X
- **Line 9:** Randomly selects k unique indices from n data points without replacement
- **Line 10:** Returns the data points at those indices as initial centroids, converted to float64

##### `_assign_labels(X, centroids)` (lines 12-18)
- **Purpose:** Assigns each data point to the nearest centroid (cluster assignment step)
- **Parameters:**
  - `X`: Data array (n, d)
  - `centroids`: Current centroid positions (k, d)
- **Line 16:** Computes squared Euclidean distances between all data points and all centroids
  - Uses broadcasting: `X[:, None, :]` expands X to (n, 1, d), `centroids[None, :, :]` expands to (1, k, d)
  - Subtracts and squares element-wise, then sums along the last axis to get (n, k) distance matrix
- **Line 17:** Finds the index of the nearest centroid for each data point using `argmin()`
- **Line 18:** Returns array of labels (cluster assignments) of shape (n,)

##### `_compute_centroids(X, labels, k)` (lines 20-30)
- **Purpose:** Updates centroid positions to the mean of their assigned data points
- **Parameters:**
  - `X`: Data array (n, d)
  - `labels`: Cluster assignments (n,)
  - `k`: Number of clusters
- **Line 21:** Gets the number of features (d) from the second dimension of X
- **Line 22:** Initializes array of zeros for new centroids, shape (k, d)
- **Lines 23-29:** Loops through each cluster:
  - **Line 24:** Gets all data points assigned to this cluster
  - **Line 25:** Checks if cluster has no members (empty cluster)
  - **Lines 26-27:** If empty, reinitializes centroid with a random data point to prevent dead clusters
  - **Line 29:** Otherwise, sets centroid to the mean (average) of all members
- **Line 30:** Returns the updated centroids array

##### `kmeans(X, k=3, max_iter=100, tol=1e-4, seed=0)` (lines 32-46)
- **Purpose:** Main K-means clustering algorithm implementation
- **Parameters:**
  - `X`: Data array (n, d)
  - `k`: Number of clusters (default 3)
  - `max_iter`: Maximum iterations (default 100)
  - `tol`: Convergence tolerance - stops if centroid shift is less than this (default 1e-4)
  - `seed`: Random seed (default 0)
- **Returns:** `labels` (n,) - cluster assignments, `centroids` (k, d) - final centroid positions
- **Line 38:** Initializes centroids using `_init_centroids()`
- **Lines 39-45:** Main iteration loop:
  - **Line 40:** Assigns each point to nearest centroid
  - **Line 41:** Computes new centroid positions
  - **Line 42:** Calculates the Euclidean norm (distance) of centroid movement
  - **Line 43:** Updates centroids to new positions
  - **Lines 44-45:** Checks if centroids moved less than tolerance - if yes, algorithm converged, break early
- **Line 46:** Returns final labels and centroids

##### `kmeans_classify(image_arr, k=3, max_iter=100, tol=1e-4, random_seed=0)` (lines 48-106)
- **Purpose:** Classifies an image into water, vegetation, and other classes using K-means
- **Parameter:** `image_arr` - 3D array (H, W, 3) with channels [Green, Red, NIR] normalized 0-1
- **Returns:** 2D classified map (H, W) with values: 1=water, 2=vegetation, 3=other
- **Line 57:** Gets image dimensions: height (H), width (W), channels (C)
- **Line 58:** Asserts that image has at least 3 channels
- **Lines 63-65:** Extracts individual bands from the 3D array:
  - **Line 63:** Extracts green band (channel 0) and converts to float64
  - **Line 64:** Extracts red band (channel 1) and converts to float64
  - **Line 65:** Extracts NIR band (channel 2) and converts to float64
- **Line 66:** Sets small epsilon to prevent division by zero
- **Line 67:** Calculates NDVI (Normalized Difference Vegetation Index) for each pixel
- **Line 68:** Calculates NDWI (Normalized Difference Water Index) for each pixel
- **Line 69:** Calculates brightness as average of red, green, and NIR bands
- **Line 72:** Stacks NDVI, NDWI, and brightness into a 3D feature array (H, W, 3)
- **Line 73:** Reshapes feature array to 2D (n, 3) where n = H*W (one row per pixel)
- **Line 76:** Runs K-means clustering on the feature vectors
- **Lines 82-83:** Extracts NDVI and NDWI values from cluster centroids
- **Lines 85-86:** Identifies cluster indices:
  - **Line 85:** Cluster with highest NDWI is labeled as water (water has high NDWI)
  - **Line 86:** Cluster with highest NDVI is labeled as vegetation (vegetation has high NDVI)
- **Lines 88-95:** Identifies the "other" cluster:
  - **Line 88:** Creates set of all cluster indices
  - **Line 89:** Finds remaining cluster(s) not assigned to water or vegetation
  - **Lines 90-91:** If exactly one remaining cluster, assigns it as "other"
  - **Lines 92-95:** Handles edge case if water and vegetation heuristics pick the same cluster, uses remaining cluster by brightness
- **Lines 98-101:** Creates mapping dictionary from cluster index to class label:
  - **Line 99:** Maps water cluster index to class label 1
  - **Line 100:** Maps vegetation cluster index to class label 2
  - **Line 101:** Maps other cluster index to class label 3
- **Line 104:** Applies mapping to convert cluster labels to class labels using vectorized function
- **Line 105:** Reshapes the 1D label array back to 2D image shape (H, W) and converts to int32
- **Line 106:** Returns the classified map

---

### 5. src/area_calculator.py - Water Area Calculation

**Libraries Used:**
- `numpy` (np): For array operations and counting pixels

#### Functions:

##### `compute_water_area_km2(classified_map, pixel_resolution_m=30.0)` (lines 5-15)
- **Purpose:** Calculates the total water area in square kilometers from a classified map
- **Parameters:**
  - `classified_map`: 2D numpy array where water pixels are labeled as 1
  - `pixel_resolution_m`: Size of each pixel in meters (default 30 for Landsat)
- **Returns:** Water area in square kilometers (float)
- **Line 11:** Counts the number of pixels with value 1 (water class) using boolean comparison and sum
- **Line 12:** Calculates area of a single pixel in square meters (pixel_size × pixel_size)
- **Line 13:** Calculates total water area in square meters (number of pixels × pixel area)
- **Line 14:** Converts square meters to square kilometers by dividing by 1,000,000 (1e6)
- **Line 15:** Returns the water area in square kilometers

---

### 6. requirements.txt - Project Dependencies

**Purpose:** Lists all external Python packages required for the project

- **Line 1:** `numpy` - Numerical computing library for array operations
- **Line 2:** `matplotlib` - Plotting and visualization library for creating charts and displaying images
- **Line 3:** `Pillow` - Python Imaging Library for image processing (used as fallback for TIFF reading)
- **Line 4:** `tkintertable` - Note: This package is listed but not actually used in the codebase
- **Line 5:** `rasterio` - Library for reading and writing geospatial raster data (GeoTIFF files)
- **Line 7:** Comment showing the pip install command to install all dependencies

---

### 7. src/__init__.py - Package Initialization

**Purpose:** Makes the `src` directory a Python package (allows imports like `from src.gui import ...`)

- **Line 1:** Empty file - Python recognizes the directory as a package, no initialization code needed

---

## Project Workflow Summary

1. **User Input:** User selects Red, Green, and NIR band files for 3 images via GUI file browsers
2. **Image Reading:** `read_image_bands()` reads three separate TIFF files and stacks them into a 3D array
3. **Feature Extraction:** For each pixel, computes NDVI, NDWI, and brightness features
4. **Clustering:** K-means algorithm groups pixels into 3 clusters based on spectral similarity
5. **Class Mapping:** Clusters are mapped to semantic classes (water, vegetation, other) based on centroid features
6. **Area Calculation:** Water pixels are counted and converted to square kilometers using pixel resolution
7. **Visualization:** Classified images and bar chart are displayed in the GUI with a legend

---

## Key Algorithms and Concepts

### K-Means Clustering
- **Initialization:** Randomly selects K data points as initial centroids
- **Assignment:** Each pixel is assigned to the nearest centroid (Euclidean distance)
- **Update:** Centroids are moved to the mean position of their assigned pixels
- **Convergence:** Algorithm stops when centroids stop moving significantly or max iterations reached

### Spectral Indices
- **NDVI (Normalized Difference Vegetation Index):** (NIR - Red) / (NIR + Red)
  - High values indicate healthy vegetation
- **NDWI (Normalized Difference Water Index):** (Green - NIR) / (Green + NIR)
  - High values indicate water presence

### Classification Strategy
- Uses NDVI, NDWI, and brightness as features for clustering
- After clustering, maps clusters to classes:
  - Highest NDWI → Water
  - Highest NDVI → Vegetation
  - Remaining → Other

---

## Library Usage Summary

- **tkinter/ttk:** GUI framework for creating windows, buttons, labels, file dialogs
- **threading:** Runs classification in background thread to prevent GUI freezing
- **numpy:** Array operations, mathematical computations, reshaping, masking
- **matplotlib:** Creating plots, subplots, bar charts, image display, legends
- **rasterio:** Reading GeoTIFF files and extracting spectral bands
- **PIL/Pillow:** Fallback image reader (not actively used in current implementation)

---

## Notes

- The project uses a manual K-means implementation (no sklearn)
- All processing happens in memory - no files are saved
- GUI is responsive with proper layout management
- Error handling is implemented throughout for robust operation
- The classification uses spectral indices (NDVI, NDWI) as features rather than raw band values

