# DeepCSAT_Final_Project

This package contains a Colab-ready notebook for end-to-end model development
and a FastAPI `app.py` for local deployment.

## Folder structure
- `data/` → place your `eCommerce_Customer_support_data.csv` here
- `notebooks/DeepCSAT_Full_Model.ipynb` → run this in Colab to train and save model
- `src/app.py` → FastAPI deployment app (loads model from `models/`)
- `models/` → model saved after training
- `processed/` → predictions saved after training
- `requirements.txt` → dependencies

## Usage (Colab)
1. Upload this folder to Google Drive.
2. Open `notebooks/DeepCSAT_Full_Model.ipynb` in Colab.
3. Run all cells — it will train models, compare them, save best model to `models/csat_pipeline.pkl`.

## Usage (Local Deployment)
1. Copy the trained model (`csat_pipeline.pkl`) from Colab to `models/`.
2. Run in terminal:
   ```bash
   pip install -r requirements.txt
   uvicorn src.app:app --reload --port 8000
   ```
3. Open `http://127.0.0.1:8000/docs` in browser to test API.