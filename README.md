# Rice Production Estimation from Drone Imagery

Code for the manuscript *"Rice Production Estimation from Drone Imagery Validated with Farmer-Reported Harvest Data"* (IJ-AI, manuscript ID 31320).

## Pipeline

1. **Image metadata and pre-processing**: EXIF/XMP reading, duplicate removal by content fingerprint, near-nadir selection (gimbal −80° to −90°, height ≥ 30 m), and ground sampling distance (GSD) from the relative flight height.
2. **Plot segmentation**: YOLOv8s-seg trained on polygon labels, with rule-based label cleaning (`scripts/clean_labels.py`).
3. **Growth-phase classification**: YOLOv8s-cls with seven crop-age classes, evaluated with stratified five-fold cross-validation.
4. **Feature extraction**: plot area (pixels × GSD²) and four RGB vegetation indices (ExG, VARI, GLI, NGRDI) averaged inside each plot mask.
5. **Field validation**: plot area and area-based production estimate compared with farmer-reported harvest data.
6. **Regression**: RF, SVR, KNN, polynomial LR, and Conv1D CNN with nested cross-validation (5-fold × 10 repeats, inner 3-fold grid search).

## How to run

Open `notebooks/pipeline_rice_production.ipynb` in Google Colab (GPU runtime), mount Google Drive, set `ROOT` to the data folder, and run the sections A–G in order. All outputs (CSV tables, figures, and trained models) are written to `ROOT/hasil/`.

## Data

The drone images and harvest records are available from the corresponding author on reasonable request. The expected folder structure and the harvest spreadsheet format (`panen.xlsx`, sheets `petak_lapangan` and `pencocokan`) are described in the notebook.

## Environment

Python 3.13, Ultralytics 8.4.202, PyTorch 2.11.0, scikit-learn 1.6.1, TensorFlow 2.21.0, Google Colab with an NVIDIA Tesla T4 GPU. See `requirements.txt`.

## License

MIT License.
