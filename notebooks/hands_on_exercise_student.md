# Soil Spectroscopy — Student Version
**Author:** Sadia Mitu  
**University of Nebraska–Lincoln**   
**Module:** Soil Spectroscopy  

This notebook is the **student version** of the hands-on exercise.  
Sections with `TODO` are intentionally incomplete — fill them in to complete the exercise!



```python
# ================================
# 1. Setup
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models

```


```python
# ================================
# 2. Load and Inspect Data
# ================================

# TODO: Load the dataset 'MIR_spectra.csv' into a pandas DataFrame named 'df'
# Hint: Use pd.read_csv()

# TODO: Display the first few rows of the dataset


# Identify spectral vs. target columns
def is_wl(c):
    try:
        float(c)
        return True
    except:
        return False

# TODO: Define 'wave_cols' as spectral columns and 'meta_cols' as metadata/targets

print("Metadata/Property Columns:", meta_cols)
print("Spectral Columns:", len(wave_cols))

```


```python
# ================================
# 3. Exploratory Data Analysis
# ================================

# TODO: Compute and display summary statistics for 'pH', and 'Clay (%)'

# TODO: Plot histograms for each target variable to understand distribution

```


```python
# ================================
# 4. Spectral Preprocessing
# ================================

from scipy.signal import savgol_filter

# Implement Standard Normal Variate (SNV)
# TODO: Complete the function below
def snv(X):
    """Standard Normal Variate normalization."""
    # Your code here
    pass

# TODO: Apply SNV to spectral columns

# TODO: Apply Savitzky–Golay smoothing (window_length=11, polyorder=2)

```


```python
# ================================
# 5. Dimensionality Reduction (PCA)
# ================================

# TODO: Perform PCA on preprocessed spectra
# Hint: Use PCA(n_components=3)

# TODO: Plot the first two principal components colored by a target variable (e.g., pH)

```


```python
# ================================
# 6. Modeling — PLSR (Baseline)
# ================================

# TODO: Split data into training/testing sets (70/30)

# TODO: Train a PLSRegression model (n_components=10)

# TODO: Predict and compute R² and RMSE

# TODO: Print the results clearly

```


```python
# ================================
# 7. Modeling — MLP (Deep Learning)
# ================================

# Build a simple neural network for regression
# TODO: Complete the model architecture
# model = models.Sequential([
    # Example: layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    # Your code here
# ])

# TODO: Compile the model using optimizer='adam' and loss='mse'

# TODO: Train the model for 100 epochs with batch_size=16

# TODO: Evaluate model performance and compute R² and RMSE on test data

```


```python
# ================================
# 8. Comparison & Discussion
# ================================

# TODO: Create a comparison table showing PLSR vs. MLP performance

# TODO: Generate scatter plots of predicted vs. observed values for each property

```

---
## Reflection Questions
1. Why might MLPs outperform PLSR for certain soil properties?  
2. What preprocessing steps most improved your model’s performance?  
3. How would you extend this workflow for NIR or VIS spectra?



```python

```
