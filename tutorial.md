# 🌾 Tutorial Overview — AI in Soil Spectroscopy
Welcome to the *AI in Agriculture* hands-on module!  
In this tutorial, you’ll learn how to use **machine learning** to predict **Soil Organic Carbon (SOC)** from soil spectral data.

---

## 🎯 Objectives
By the end of this exercise you will:
1. Understand how soil spectra represent soil composition.  
2. Visualize raw and preprocessed spectra.  
3. Apply basic preprocessing (SNV + Savitzky–Golay).  
4. Build a simple **Partial Least Squares Regression (PLSR)** model.  
5. Evaluate predictions and interpret results.

---

## 📁 Dataset Description
File: `soil_spectra_teaching.csv`  
Each row is a soil sample.

| Type | Columns | Description |
|------|----------|-------------|
| Spectral | `wl_400 ... wl_2500` | Reflectance at each wavelength (nm) |
| Target | `SOC` | Soil Organic Carbon (%) |
| Optional | `pH`, `Clay`, etc. | Other soil properties |

---




```python
# 🔧 Load packages and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("MIR_scans_for_neo_outlier.csv")
df = pd.read_csv(DATA_PATH)
df.head()

```





  <div id="df-5e953839-1417-4f3d-ad2a-986f3cd60163" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>smp_id</th>
      <th>ph_h2o</th>
      <th>eoc_tot_c</th>
      <th>clay_tot_psa</th>
      <th>599.766</th>
      <th>601.695</th>
      <th>603.623</th>
      <th>605.552</th>
      <th>607.48</th>
      <th>609.409</th>
      <th>...</th>
      <th>3936.069</th>
      <th>3937.998</th>
      <th>3939.926</th>
      <th>3941.855</th>
      <th>3943.783</th>
      <th>3945.712</th>
      <th>3947.64</th>
      <th>3949.569</th>
      <th>3951.497</th>
      <th>3953.426</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>124566.0</td>
      <td>5.90</td>
      <td>1.4</td>
      <td>34.1</td>
      <td>1.76600</td>
      <td>1.77350</td>
      <td>1.77950</td>
      <td>1.78525</td>
      <td>1.79150</td>
      <td>1.7985</td>
      <td>...</td>
      <td>0.418650</td>
      <td>0.418500</td>
      <td>0.416825</td>
      <td>0.414450</td>
      <td>0.413375</td>
      <td>0.413300</td>
      <td>0.412550</td>
      <td>0.412025</td>
      <td>0.412600</td>
      <td>0.413150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>124573.0</td>
      <td>4.80</td>
      <td>2.0</td>
      <td>19.1</td>
      <td>1.62925</td>
      <td>1.63725</td>
      <td>1.64725</td>
      <td>1.65800</td>
      <td>1.67025</td>
      <td>1.6845</td>
      <td>...</td>
      <td>0.383575</td>
      <td>0.383775</td>
      <td>0.382475</td>
      <td>0.380525</td>
      <td>0.379825</td>
      <td>0.380025</td>
      <td>0.379425</td>
      <td>0.379100</td>
      <td>0.379800</td>
      <td>0.380550</td>
    </tr>
    <tr>
      <th>2</th>
      <td>124574.0</td>
      <td>4.94</td>
      <td>1.0</td>
      <td>34.9</td>
      <td>1.76025</td>
      <td>1.76925</td>
      <td>1.77700</td>
      <td>1.78550</td>
      <td>1.79575</td>
      <td>1.8080</td>
      <td>...</td>
      <td>0.411900</td>
      <td>0.411700</td>
      <td>0.410025</td>
      <td>0.407750</td>
      <td>0.406800</td>
      <td>0.406700</td>
      <td>0.405875</td>
      <td>0.405350</td>
      <td>0.405900</td>
      <td>0.406475</td>
    </tr>
    <tr>
      <th>3</th>
      <td>124584.0</td>
      <td>5.77</td>
      <td>3.4</td>
      <td>31.8</td>
      <td>1.75225</td>
      <td>1.75700</td>
      <td>1.76025</td>
      <td>1.76275</td>
      <td>1.76625</td>
      <td>1.7705</td>
      <td>...</td>
      <td>0.425150</td>
      <td>0.425350</td>
      <td>0.424050</td>
      <td>0.422150</td>
      <td>0.421475</td>
      <td>0.421650</td>
      <td>0.421050</td>
      <td>0.420625</td>
      <td>0.421325</td>
      <td>0.422025</td>
    </tr>
    <tr>
      <th>4</th>
      <td>124585.0</td>
      <td>5.65</td>
      <td>1.1</td>
      <td>44.4</td>
      <td>1.80775</td>
      <td>1.81500</td>
      <td>1.82275</td>
      <td>1.83150</td>
      <td>1.84050</td>
      <td>1.8495</td>
      <td>...</td>
      <td>0.454700</td>
      <td>0.454600</td>
      <td>0.453000</td>
      <td>0.450750</td>
      <td>0.449675</td>
      <td>0.449550</td>
      <td>0.448675</td>
      <td>0.448125</td>
      <td>0.448625</td>
      <td>0.449150</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1744 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5e953839-1417-4f3d-ad2a-986f3cd60163')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5e953839-1417-4f3d-ad2a-986f3cd60163 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5e953839-1417-4f3d-ad2a-986f3cd60163');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-9612b34d-2691-4e20-815a-0b7d06f97302">
      <button class="colab-df-quickchart" onclick="quickchart('df-9612b34d-2691-4e20-815a-0b7d06f97302')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-9612b34d-2691-4e20-815a-0b7d06f97302 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>






# 📊 Basic info


```python

print("Samples:", df.shape[0], " | Columns:", df.shape[1])

spec_cols = [c for c in df.columns if str(c).replace('.', '', 1).isdigit()]
print("Spectral columns:", len(spec_cols))
print("Other columns:", [c for c in df.columns if c not in spec_cols])


```

    Samples: 343  | Columns: 1744
    Spectral columns: 1740
    Other columns: ['smp_id', 'ph_h2o', 'eoc_tot_c', 'clay_tot_psa']




# 🌈 Plot Raw Spectra


```python
import matplotlib.pyplot as plt
import numpy as np

# 1. Identify spectral columns (numeric names)
spec_cols = [c for c in df.columns if str(c).replace('.', '', 1).isdigit()]

# 2. Extract wavelengths and spectral matrix
wavelengths = np.array(spec_cols, dtype=float)
X = df[spec_cols].values.astype(float)

# 3. Plot up to 50 spectra
plt.figure(figsize=(9,5))
for i in range(min(50, X.shape[0])):   # plot up to 50 random samples
    plt.plot(wavelengths, X[i, :], alpha=0.5)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (%)")
plt.title("Raw MIR Spectra (Reflectance vs Wavelength)")
plt.grid(alpha=0.3)
plt.show()

```


    
![png](tutorial_files/tutorial_7_0.png)
    


# ⚙️ Preprocessing — SNV + Savitzky–Golay filter


```python

from scipy.signal import savgol_filter

def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)

X_raw = df[spec_cols].to_numpy(float)
X_snv = snv(X_raw)
window = 11 if X_snv.shape[1] >= 21 else (X_snv.shape[1]//2*2+1)
X_sg = savgol_filter(X_snv, window_length=window, polyorder=2, deriv=1, axis=1)

```

# 📉 Plot Preprocessed Spectra


```python

plt.figure(figsize=(8,5))
for i in range(min(50, len(df))):
    plt.plot(wavelengths, X_sg[i], alpha=0.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Processed Reflectance (1st Derivative)")
plt.title("Preprocessed Soil Spectra (SNV + SG)")
plt.show()

```


    
![png](tutorial_files/tutorial_11_0.png)
    




# 🔢 Split data and fit a simple PLSR model


```python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Drop rows where SOC is missing
df = df.dropna(subset=["eoc_tot_c"]).reset_index(drop=True)
print("After dropping null SOC rows:", df.shape)

# 2️⃣ Detect spectral columns again (from cleaned df)
spec_cols = [c for c in df.columns if str(c).replace('.', '', 1).isdigit()]

# 3️⃣ Extract spectra (X) and target (y)
X = df[spec_cols].values.astype(float)
y = df["eoc_tot_c"].values.astype(float)

# 4️⃣ Check they match
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for PLS)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Fit PLSR model
pls = PLSRegression(n_components=10)  # you can tune n_components
pls.fit(X_train_s, y_train)

# Predictions
y_pred = pls.predict(X_test_s).ravel()

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")



```

    After dropping null SOC rows: (109, 1744)
    X shape: (109, 1740)
    y shape: (109,)
    R² = 0.920
    RMSE = 0.388




# 📈 Scatter (Parity) Plot — Observed vs Predicted SOC


```python

plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='none')
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], 'k--')
plt.xlabel("Observed SOC (%)")
plt.ylabel("Predicted SOC (%)")
plt.title(f"PLSR Prediction — SOC (R²={r2:.2f})")
plt.tight_layout()
plt.show()

```


    
![png](tutorial_files/tutorial_17_0.png)
    



```python

```


```python
from google.colab import drive
drive.mount('/content/drive')

```


```python
!jupyter nbconvert --to markdown "/content/drive/My Drive/Colab Notebooks/tutorial.ipynb"



```

    [NbConvertApp] WARNING | pattern '/content/drive/My Drive/Colab Notebooks/tutorial.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

