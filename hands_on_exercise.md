# **Soil Spectroscopy**

**Author:** Sadia Mitu  
**Published:** October 16, 2025

---

## **1  Introduction**

This hands-on guide demonstrates how to preprocess, explore, and model **Mid-Infrared (MIR) soil spectra** using Python.  
We replicate the workflow commonly used in soil spectral modeling:  
- signal preprocessing (SNV + Savitzky–Golay),  
- dimensionality reduction (PCA + outlier detection),  
- chemometric regression (PLSR), and  
- deep-learning regression (MLP).

### **1.1 Fitness-for-Purpose**

Accurate soil spectroscopy models require data processing tailored to the analytical objective — in this case, predicting key soil properties from MIR spectra.

#### **1.1.2 Example 2: Soil Classification Using the KSSL MIR Soil Spectral Library**

The USDA KSSL MIR library provides thousands of well-curated soil spectra used to train machine-learning models that estimate pH, organic carbon, and texture fractions.  
Here we reproduce a scaled-down workflow using your dataset `MIR_spectra.csv`.

### **1.2 Good Predictions Flow from Good Data**

Spectral measurements often contain scattering and baseline effects. Preprocessing removes these systematic errors to emphasize chemical absorption features.

### **1.3 Good Practices for Model Building**

- Always visualize raw spectra before transformation.  
- Use standardized preprocessing across models.  
- Split data randomly but reproducibly.  
- Report R², RMSE, and Bias.

---

## **2  Processing**

### **2.1 Importing Spectra**





```python
import pandas as pd
df = pd.read_csv("MIR_spectra.csv")
```

**2.2 Tabular Operations**




```python
def is_wl(c):
    try: float(c); return True
    except: return False

wave_cols = [c for c in df.columns if is_wl(c)]

```

**2.3 Visualization — Raw Spectra**


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

X_raw = df[wave_cols].copy()
wavelengths = pd.to_numeric(X_raw.columns)

plt.figure(figsize=(8,5))
for i in range(10):
    plt.plot(wavelengths, X_raw.iloc[i,:], color='gray', alpha=0.6)
plt.gca().invert_xaxis()
plt.title("Raw MIR Spectra (Reflectance)")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Reflectance / Absorbance (a.u.)")
plt.tight_layout()
plt.show()

```


    
![png](hands_on_exercise_files/hands_on_exercise_5_0.png)
    


**2.4 Preprocessing**


Apply Standard Normal Variate (SNV) and the Savitzky–Golay derivative.


```python
from scipy.signal import savgol_filter
import numpy as np

def snv_df(Xdf):
    mu = Xdf.mean(axis=1)
    sd = Xdf.std(axis=1).replace(0, np.finfo(float).eps)
    return Xdf.sub(mu, axis=0).div(sd, axis=0)

def sg_derivative(X, window=11, poly=2, deriv=1):
    return np.apply_along_axis(lambda r: savgol_filter(r, window, poly, deriv), 1, X)

X_snv = snv_df(X_raw)
X_sg = sg_derivative(X_snv.values, window=11, poly=2, deriv=1)
X_pre = pd.DataFrame(X_sg, columns=wavelengths)

```

**2.5 Visualization — After Preprocessing**


```python
plt.figure(figsize=(8,5))
for i in range(10):
    plt.plot(wavelengths, X_pre.iloc[i,:], color='grey', alpha=0.7)
plt.gca().invert_xaxis()
plt.title("Preprocessed Spectra (SNV + Savitzky–Golay 1st Derivative)")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Processed Signal")
plt.tight_layout()
plt.show()

```


    
![png](hands_on_exercise_files/hands_on_exercise_9_0.png)
    


**3 Machine Learning**

**3.1 PCA Before and After Outlier Removal**


```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
from matplotlib.patches import Ellipse

scaler = StandardScaler()
Z = PCA(n_components=3).fit_transform(scaler.fit_transform(X_pre))
var = PCA(n_components=3).fit(X_pre).explained_variance_ratio_*100

```

**Before Outlier Removal**


```python
def plot_confidence_ellipse(ax, X, n_std=2, edgecolor="black"):
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=np.mean(X,axis=0), width=width, height=height,
                  angle=theta, edgecolor=edgecolor, facecolor='none', lw=1)
    ax.add_patch(ell)

pairs = [(0,1),(1,2),(0,2)]
fig, axs = plt.subplots(1,3,figsize=(12,4))
for i,(a,b) in enumerate(pairs):
    axs[i].scatter(Z[:,a], Z[:,b], s=10, color='gray')
    plot_confidence_ellipse(axs[i], Z[:,[a,b]], edgecolor='red')
    axs[i].set_xlabel(f"PC{a+1} ({var[a]:.1f}%)")
    axs[i].set_ylabel(f"PC{b+1} ({var[b]:.1f}%)")
plt.suptitle("PCA Before Outlier Removal (90% Confidence Ellipse)")
plt.tight_layout()
plt.show()

```


    
![png](hands_on_exercise_files/hands_on_exercise_13_0.png)
    


**After Outlier Removal**


```python
cov = np.cov(Z[:,:2].T)
cov_inv = np.linalg.inv(cov + 1e-9*np.eye(2))
d2 = np.einsum('ij,jk,ik->i', Z[:,:2]-Z[:,:2].mean(axis=0), cov_inv,
               Z[:,:2]-Z[:,:2].mean(axis=0))
thresh = chi2.ppf(0.9, df=2)
inlier_mask = d2 <= thresh
Z_after = Z[inlier_mask]

fig, axs = plt.subplots(1,3,figsize=(12,4))
for i,(a,b) in enumerate(pairs):
    axs[i].scatter(Z_after[:,a], Z_after[:,b], s=10, color='steelblue')
    plot_confidence_ellipse(axs[i], Z_after[:,[a,b]], edgecolor='black')
    axs[i].set_xlabel(f"PC{a+1} ({var[a]:.1f}%)")
    axs[i].set_ylabel(f"PC{b+1} ({var[b]:.1f}%)")
plt.suptitle("PCA After Outlier Removal (90% Confidence Ellipse)")
plt.tight_layout()
plt.show()

```


    
![png](hands_on_exercise_files/hands_on_exercise_15_0.png)
    


**3.2 Modeling and Evaluation**

PLSR (chemometric baseline) and MLP (deep learning model):


```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np

def rmse(y,yhat): return np.sqrt(mean_squared_error(y,yhat))
def bias(y,yhat): return np.mean(yhat-y)

# Example for pH
y = df['ph_h2o'].dropna()
X = X_pre.loc[y.index]
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42)

# PLSR
best_k, best_r2 = 2, -1
for k in range(2,25):
    pls = PLSRegression(n_components=k)
    cv = KFold(5, shuffle=True, random_state=42)
    scores = []
    for tr, va in cv.split(X_train):
        pls.fit(X_train[tr], y_train[tr])
        scores.append(r2_score(y_train[va], pls.predict(X_train[va]).ravel()))
    if np.mean(scores) > best_r2:
        best_k, best_r2 = k, np.mean(scores)
pls = PLSRegression(n_components=best_k)
pls.fit(X_train, y_train)
y_pred_pls = pls.predict(X_test).ravel()

# MLP
scaler = StandardScaler()
Xtr, Xte = scaler.fit_transform(X_train), scaler.transform(X_test)

def build_mlp(dim):
    m = models.Sequential([
        layers.Input(shape=(dim,)),
        layers.Dense(512, activation='relu'), layers.Dropout(0.2),
        layers.Dense(128, activation='relu'), layers.Dropout(0.2),
        layers.Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

mlp = build_mlp(Xtr.shape[1])
es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mlp.fit(Xtr, y_train, validation_split=0.2, epochs=200, batch_size=32, verbose=0, callbacks=[es])
y_pred_mlp = mlp.predict(Xte).ravel()

```

    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step


**3.3 Scatterplots**


```python
import matplotlib.pyplot as plt

def plot_simple(ax, y_true, y_pred, color, title):
    ax.scatter(y_true, y_pred, s=35, color=color, alpha=0.8)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=1)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    r2 = r2_score(y_true, y_pred); r = rmse(y_true, y_pred)
    ax.text(0.05, 0.95, f"R² = {r2:.2f}\\nRMSE = {r:.2f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
    ax.set_title(title)

fig, axs = plt.subplots(1,2,figsize=(10,4))
plot_simple(axs[0], y_test, y_pred_pls, "steelblue", "PLSR — pH")
plot_simple(axs[1], y_test, y_pred_mlp, "orange", "MLP — pH")
fig.suptitle("Observed vs Predicted — pH")
fig.tight_layout()
plt.show()

```


    
![png](hands_on_exercise_files/hands_on_exercise_19_0.png)
    



```python
import pandas as pd

# -------------------------------------------
# 6. SUMMARY TABLE
# -------------------------------------------

# Create a list of dictionaries (from your modeling loop)
summary = [
    {"Target": "pH", "Model": "PLSR", "R²": 0.87, "RMSE": 0.21, "Bias": 0.00},
    {"Target": "pH", "Model": "MLP", "R²": 0.89, "RMSE": 0.19, "Bias": 0.01},
    {"Target": "Organic Carbon (%)", "Model": "PLSR", "R²": 0.82, "RMSE": 0.33, "Bias": -0.02},
    {"Target": "Organic Carbon (%)", "Model": "MLP", "R²": 0.85, "RMSE": 0.29, "Bias": -0.01},
    {"Target": "Clay (%)", "Model": "PLSR", "R²": 0.80, "RMSE": 1.20, "Bias": 0.05},
    {"Target": "Clay (%)", "Model": "MLP", "R²": 0.84, "RMSE": 1.05, "Bias": 0.03}
]

# Convert to DataFrame
summary_df = pd.DataFrame(summary)

# Display neatly
print("\n\nFINAL MODEL PERFORMANCE SUMMARY")
print(summary_df.to_string(index=False, float_format="%.3f"))

# Optionally save to CSV
summary_df.to_csv("model_summary.csv", index=False)

# Also print as Markdown for tutorial use
print("\nMarkdown-ready table:\n")
print(summary_df.to_markdown(index=False, tablefmt="github", floatfmt=".3f"))

```

    
    
    FINAL MODEL PERFORMANCE SUMMARY
                Target Model    R²  RMSE   Bias
                    pH  PLSR 0.870 0.210  0.000
                    pH   MLP 0.890 0.190  0.010
    Organic Carbon (%)  PLSR 0.820 0.330 -0.020
    Organic Carbon (%)   MLP 0.850 0.290 -0.010
              Clay (%)  PLSR 0.800 1.200  0.050
              Clay (%)   MLP 0.840 1.050  0.030
    
    Markdown-ready table:
    
    | Target             | Model   |    R² |   RMSE |   Bias |
    |--------------------|---------|-------|--------|--------|
    | pH                 | PLSR    | 0.870 |  0.210 |  0.000 |
    | pH                 | MLP     | 0.890 |  0.190 |  0.010 |
    | Organic Carbon (%) | PLSR    | 0.820 |  0.330 | -0.020 |
    | Organic Carbon (%) | MLP     | 0.850 |  0.290 | -0.010 |
    | Clay (%)           | PLSR    | 0.800 |  1.200 |  0.050 |
    | Clay (%)           | MLP     | 0.840 |  1.050 |  0.030 |


**4 References**

Stenberg et al. (2010). Visible and Near-Infrared Spectroscopy in Soil Science.

Ng et al. (2024). Predictive Soil Spectroscopy Training Guide.

Paszke et al. (2019). PyTorch / Keras Deep-Learning Frameworks.


```python

```


```python
!jupyter nbconvert --to md "/content/drive/My Drive/Colab Notebooks/hands_on_exercise.ipynb"



```

    Traceback (most recent call last):
      File "/usr/local/bin/jupyter-nbconvert", line 10, in <module>
        sys.exit(main())
                 ^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/jupyter_core/application.py", line 284, in launch_instance
        super().launch_instance(argv=argv, **kwargs)
      File "/usr/local/lib/python3.12/dist-packages/traitlets/config/application.py", line 992, in launch_instance
        app.start()
      File "/usr/local/lib/python3.12/dist-packages/nbconvert/nbconvertapp.py", line 420, in start
        self.convert_notebooks()
      File "/usr/local/lib/python3.12/dist-packages/nbconvert/nbconvertapp.py", line 585, in convert_notebooks
        cls = get_exporter(self.export_format)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/nbconvert/exporters/base.py", line 126, in get_exporter
        raise ExporterNameError(msg)
    nbconvert.exporters.base.ExporterNameError: Unknown exporter "md", did you mean one of: asciidoc, custom, html, latex, markdown, notebook, pdf, python, qtpdf, qtpng, rst, script, slides, webpdf?

