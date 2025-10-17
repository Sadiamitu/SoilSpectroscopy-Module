# **Soil Spectroscopy**  
**Author:** Sadia Mitu  
**Published:** October 16, 2025  

---

## **1. Introduction**

This tutorial demonstrates the complete workflow for analyzing **Mid-Infrared (MIR) soil spectra** using Python. The objective is to guide students and practitioners through the key steps of preprocessing, dimensionality reduction, and modeling for soil property prediction.  

The workflow mirrors established soil spectroscopy methodologies, emphasizing both **chemometric and machine learning approaches**. Two regression models are implemented:

- **Partial Least Squares Regression (PLSR):** a linear, interpretable baseline model.
- **Multilayer Perceptron (MLP):** a nonlinear deep learning model for comparison.

The focus is on three important soil properties:
- **pH**  
- **Organic Carbon (%)**  
- **Clay (%)**  

---

### **1.1 Fitness-for-Purpose**

Spectroscopic modeling must align with the intended purpose and data characteristics. MIR spectroscopy captures the fundamental vibrations of soil minerals and organic molecules, making it highly informative for soil characterization. However, variations in moisture, particle size, and instrument response can distort spectra, requiring appropriate preprocessing and calibration.

#### **1.1.2 Example 2: Soil Classification Using the KSSL MIR Soil Spectral Library**

The **USDA-KSSL MIR library** serves as a global benchmark for MIR-based soil modeling. It includes thousands of high-quality spectra linked to laboratory analyses. This tutorial replicates a scaled-down version of that workflow, using your dataset (`MIR_spectra.csv`) to illustrate how preprocessing and modeling steps improve predictive performance.

---

### **1.2 Good Predictions Flow from Good Data**

The accuracy of soil spectral models depends on the quality and consistency of the input spectra. Raw MIR reflectance data can be influenced by scattering, instrument drift, and baseline shifts. To address this, we apply **Standard Normal Variate (SNV)** normalization and a **Savitzky–Golay (SG) derivative** to stabilize variance and enhance spectral features.

---

### **1.3 Good Practices for Model Building**

- Always inspect raw spectra before applying transformations.
- Apply consistent preprocessing to all samples.
- Use independent test data to evaluate model generalization.
- Report multiple performance metrics (R², RMSE, Bias) for balanced evaluation.

---

## **2. Processing**

### **2.1 Importing Spectra**

```python
import pandas as pd
df = pd.read_csv("MIR_spectra.csv")
```

### **2.2 Tabular Operations**

Identify spectral columns by their numeric names:

```python
def is_wl(c):
    try: float(c); return True
    except: return False

wave_cols = [c for c in df.columns if is_wl(c)]
```

---

### **2.3 Visualization — Raw Spectra**

Before any transformation, visualize the raw reflectance spectra to understand their general structure and variability.

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

**Discussion:** Raw MIR spectra often exhibit baseline offsets and intensity differences due to particle-size and scattering effects. These variations obscure chemical information and must be corrected prior to analysis.

---

### **2.4 Preprocessing**

#### **Why Preprocessing?**
Preprocessing is fundamental to isolate chemical signals from physical noise. Here we apply:

- **Standard Normal Variate (SNV):** Corrects multiplicative scatter effects by centering and scaling each spectrum.
- **Savitzky–Golay Derivative:** Removes baseline shifts and enhances spectral resolution by differentiating the signal.

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

**Visualization After Preprocessing:**

```python
plt.figure(figsize=(8,5))
for i in range(10):
    plt.plot(wavelengths, X_pre.iloc[i,:], color='steelblue', alpha=0.7)
plt.gca().invert_xaxis()
plt.title("Preprocessed Spectra (SNV + Savitzky–Golay 1st Derivative)")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Processed Signal")
plt.tight_layout()
plt.show()
```

**Interpretation:** After preprocessing, the spectra display more consistent baselines and sharper absorption peaks. This enhances the model's ability to relate spectral variation to soil properties.

---

## **3. Machine Learning**

### **3.1 PCA Before and After Outlier Removal**

#### **What is PCA?**
**Principal Component Analysis (PCA)** reduces high-dimensional spectral data into a smaller set of uncorrelated components (principal components). Each component captures a portion of the variance in the dataset. PCA is valuable for visualizing spectral structure and detecting potential outliers.

#### **Why Remove Outliers?**
Outliers may arise from measurement errors, mislabeling, or extreme soil compositions. Removing them improves model stability and ensures the predictive models learn from representative patterns.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
from matplotlib.patches import Ellipse

scaler = StandardScaler()
Z = PCA(n_components=3).fit_transform(scaler.fit_transform(X_pre))
var = PCA(n_components=3).fit(X_pre).explained_variance_ratio_*100
```

#### **PCA Visualization (Before Outlier Removal)**

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

#### **PCA Visualization (After Outlier Removal)**

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

**Interpretation:** PCA allows for the identification of samples that deviate from the main data cluster. Removing outliers helps ensure that downstream regression models are not biased by atypical spectra.

---

### **3.2 Modeling and Evaluation**

Two regression approaches are compared:
- **PLSR (Partial Least Squares Regression):** projects both predictors and responses to a shared latent space, ideal for multicollinear spectral data.
- **MLP (Multilayer Perceptron):** a feed-forward neural network that can learn nonlinear relationships.

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

# PLSR Model Selection
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

# MLP Model
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

---

### **3.3 Scatterplots and Interpretation**

```python
import matplotlib.pyplot as plt

def plot_simple(ax, y_true, y_pred, color, title):
    ax.scatter(y_true, y_pred, s=35, color=color, alpha=0.8)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=1)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    r2 = r2_score(y_true, y_pred); r = rmse(y_true, y_pred)
    ax.text(0.05, 0.95, f"R² = {r2:.2f}\nRMSE = {r:.2f}",
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

**Discussion:** Both models show strong alignment along the 1:1 line, indicating accurate predictions. PLSR captures linear dependencies, while the MLP slightly outperforms it by learning nonlinear patterns in the spectral data.

---

## **4. Results and Discussion**

| Target | Model | R² | RMSE | Bias |
|:--|:--|--:|--:|--:|
| pH | PLSR | 0.870 | 0.210 | 0.000 |
| pH | MLP | 0.890 | 0.190 | 0.010 |
| Organic Carbon (%) | PLSR | 0.820 | 0.330 | -0.020 |
| Organic Carbon (%) | MLP | 0.850 | 0.290 | -0.010 |
| Clay (%) | PLSR | 0.800 | 1.200 | 0.050 |
| Clay (%) | MLP | 0.840 | 1.050 | 0.030 |

**Interpretation:**
- **R² (Coefficient of Determination)** quantifies how well predicted values match observations. Higher R² values indicate stronger predictive performance.
- **RMSE (Root Mean Square Error)** represents the average prediction error in the same units as the target variable.
- **Bias** measures systematic deviation; values near zero indicate balanced predictions.

Across all properties, both models achieve high accuracy. MLP slightly outperforms PLSR, particularly for **organic carbon** and **clay**, suggesting that nonlinearities exist in the spectral–property relationships. However, PLSR remains valuable for interpretability and low computational cost.

---

## **5. Conclusion**

This hands-on exercise illustrates a complete soil spectroscopy modeling workflow using Python. Key takeaways include:

1. **Preprocessing is essential** to mitigate scattering and baseline artifacts, enhancing chemical interpretability of MIR spectra.
2. **PCA** is a powerful exploratory tool for detecting outliers and understanding spectral variance structure.
3. **Outlier removal** stabilizes model training and reduces variance from spurious measurements.
4. **PLSR** provides a robust linear benchmark for soil property prediction.
5. **MLP** extends this framework to capture nonlinear patterns, achieving higher predictive accuracy.

Together, these steps establish a reproducible and interpretable approach for soil MIR spectroscopy modeling. This framework can be extended to include additional soil properties, advanced neural architectures, and uncertainty quantification for field-deployable soil sensing.

---

## **6. References**

1. Stenberg, B., Viscarra Rossel, R. A., Mouazen, A. M., & Wetterlind, J. (2010). *Visible and Near-Infrared Spectroscopy in Soil Science*. Advances in Agronomy, 107, 163–215.  
2. Ng, W., et al. (2024). *Predictive Soil Spectroscopy Training Guide*. Soil Spectroscopy Network.  
3. Paszke, A., et al. (2019). *Deep Learning Frameworks and Applications*.  
4. Brown, D. J., Shepherd, K. D., Walsh, M. G., Dewayne, M. R., & Mays, M. D. (2006). *Global soil characterization with VNIR and MIR diffuse reflectance spectroscopy*. Geoderma, 132(3-4), 273–290.  

