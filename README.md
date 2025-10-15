# AI in Agriculture: Machine Learning for Soil Spectroscopy
Hands-on module with **Python**: PLSR baseline + tiny 1D CNN for soil spectra.

## Structure
- `data/soil_spectra_teaching.csv` — teaching dataset
- `notebooks/01_plsr_baseline.ipynb` — PLSR baseline
- `notebooks/02_cnn_1d.ipynb` — simple 1D CNN (Keras)
- `docs/tutorial_instructions.md` — step-by-step lab
- `env/requirements.txt` — packages

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r env/requirements.txt
jupyter lab
```
Open the notebooks in `notebooks/` and run top-to-bottom.

## What to submit
- Metrics table + parity plot(s)
- Short memo (5–7 bullets) on findings and pitfalls
