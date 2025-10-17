<details>
<summary>Show Code</summary>


# **Soil Spectroscopy**

**Author:** Sadia Mitu       
**Email:** smitu2@huskers.unl.edu  
**Published:** October 16, 2025

---

## **1  Introduction**

This hands-on guide demonstrates how to preprocess, explore, and model Mid-Infrared (MIR) soil spectra using Python. Our goal is to predict three fundamental soil properties: pH, organic carbon (%), and clay (%) from MIR spectral data.

The workflow is built on a subset of the [USDA KSSL MIR Soil Spectral Library](https://ncsslabdatamart.sc.egov.usda.gov/), one of the world’s largest collections of high-quality soil spectra paired with laboratory analyses. Here, we use over 500 representative samples drawn from the KSSL archive to illustrate best practices in MIR spectral preprocessing, visualization, and predictive modeling.

The tutorial follows a structured, stepwise approach:

*   **Spectral preprocessing:** minimizing scattering and baseline effects using Standard Normal Variate (SNV) and Savitzky–Golay smoothing.

*   **Dimensionality reduction:** revealing spectral structure and identifying outliers through Principal Component Analysis (PCA).

*   **Chemometric modeling:** capturing linear spectral–property relationships with Partial Least Squares Regression (PLSR).

*   **Deep learning regression:** modeling nonlinear dependencies using a Multilayer Perceptron (MLP) network.

### **1.2 Importance of Data Quality and Preprocessing**

Soil spectra often include unwanted effects like scattering, baseline shifts, and instrument noise. Preprocessing helps remove these distortions, making the chemical absorption features clearer. Consistent, high-quality preprocessing is essential for building accurate and reliable soil models.

### **1.3 Principles of Robust Model Development**

Developing reliable soil spectral models requires a disciplined, transparent workflow. To ensure reproducibility and interpretability:

*   Always visualize raw spectra before transformation to detect irregularities.
*   Apply consistent preprocessing across all samples.
*   Use reproducible random splits for training and testing.
*   Report performance metrics: R², RMSE, and Bias for a comprehensive evaluation

---

## **2  Processing**

### **2.1 Tabular operation**
**Importing Data**

You can set an external folder as your working directory.
First, make sure both your notebook and the dataset file MIR_spectra.csv are saved in the same working directory (folder).
Then, run the following code to import and inspect the dataset. This code loads the MIR spectral dataset into a pandas DataFrame and displays the first few records to confirm that the data has been imported correctly.


```python
import pandas as pd

# Load dataset
df = pd.read_csv("MIR_spectra.csv")

# Display dataset structure
df.head()

```





  <div id="df-7d68b473-038a-4283-9169-3811e215e868" class="colab-df-container">
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
      <td>124566</td>
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
      <td>124573</td>
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
      <td>124574</td>
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
      <td>124584</td>
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
      <td>124585</td>
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-7d68b473-038a-4283-9169-3811e215e868')"
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
        document.querySelector('#df-7d68b473-038a-4283-9169-3811e215e868 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7d68b473-038a-4283-9169-3811e215e868');
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


    <div id="df-26f05ec2-bbb6-4e71-b53b-1b8475cc588d">
      <button class="colab-df-quickchart" onclick="quickchart('df-26f05ec2-bbb6-4e71-b53b-1b8475cc588d')"
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
            document.querySelector('#df-26f05ec2-bbb6-4e71-b53b-1b8475cc588d button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




**Importing spectra**


```python
# Identify spectral columns (wavelengths)
def is_wl(c):
    try:
        float(c)
        return True
    except:
        return False

wave_cols = [c for c in df.columns if is_wl(c)]

# Identify metadata and target columns
meta_cols = [c for c in df.columns if c not in wave_cols]
print("Metadata/Property Columns:", meta_cols)
print("Spectral Columns:", len(wave_cols))


```

    Metadata/Property Columns: ['smp_id', 'ph_h2o', 'eoc_tot_c', 'clay_tot_psa']
    Spectral Columns: 1740


### **2.2 Summary Statistics of Soil Properties**
Before modeling, we inspect basic statistics for the target variables: pH, organic carbon (%), and clay (%) to understand their spread and central tendency.


```python
# Select main target properties
targets = ["ph_h2o", "eoc_tot_c", "clay_tot_psa"]

# Summary statistics
summary_stats = df[targets].describe().T
summary_stats

```





  <div id="df-72eef855-fe25-4b46-bde2-1d1ae236e500" class="colab-df-container">
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ph_h2o</th>
      <td>517.0</td>
      <td>6.211876</td>
      <td>1.249596</td>
      <td>4.0</td>
      <td>5.2</td>
      <td>6.1</td>
      <td>7.31</td>
      <td>8.95</td>
    </tr>
    <tr>
      <th>eoc_tot_c</th>
      <td>517.0</td>
      <td>2.198646</td>
      <td>1.849549</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>1.6</td>
      <td>2.80</td>
      <td>9.90</td>
    </tr>
    <tr>
      <th>clay_tot_psa</th>
      <td>517.0</td>
      <td>19.928046</td>
      <td>12.595476</td>
      <td>0.0</td>
      <td>10.6</td>
      <td>17.9</td>
      <td>27.50</td>
      <td>66.70</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-72eef855-fe25-4b46-bde2-1d1ae236e500')"
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
        document.querySelector('#df-72eef855-fe25-4b46-bde2-1d1ae236e500 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-72eef855-fe25-4b46-bde2-1d1ae236e500');
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


    <div id="df-6b82d16b-05e0-4b0a-b169-048f3e07328d">
      <button class="colab-df-quickchart" onclick="quickchart('df-6b82d16b-05e0-4b0a-b169-048f3e07328d')"
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
            document.querySelector('#df-6b82d16b-05e0-4b0a-b169-048f3e07328d button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_69ec16c1-a5ec-4f20-9e0e-5cee58f0ecaf">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('summary_stats')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_69ec16c1-a5ec-4f20-9e0e-5cee58f0ecaf button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('summary_stats');
      }
      })();
    </script>
  </div>

    </div>
  </div>




**Property Distributions (Histograms)**

Visualizing property distributions helps detect skewness and outliers.


```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

name_map = {
    "ph_h2o": "pH",
    "eoc_tot_c": "Organic Carbon (%)",
    "clay_tot_psa": "Clay (%)"
}

targets = ["ph_h2o", "eoc_tot_c", "clay_tot_psa"]

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

for i, col in enumerate(targets):
    sns.histplot(df[col].dropna(), bins=25, kde=True, ax=axs[i], color="grey")
    axs[i].set_xlabel(name_map[col])
    axs[i].set_ylabel("Frequency" if i == 0 else "")
    axs[i].grid(False)

sns.despine()
plt.suptitle("Distributions of Soil Properties", fontsize=14)
plt.tight_layout()
plt.show()

```


    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_8_0.png)
    


**pH:** Roughly bimodal, indicating both acidic and near-neutral soils.

**Organic Carbon (%):** Strongly right-skewed, as organic-rich soils are less common.

**Clay (%):** Moderately skewed, reflecting natural variability in texture classes.


###**2.3 Preprocessing**

Raw MIR spectra often exhibit baseline offsets and intensity differences caused by particle-size variation and light-scattering effects. These physical distortions can obscure true chemical information and lead to poor model performance. Preprocessing is therefore essential to isolate chemical signals from background noise and ensure that spectral variations reflect compositional differences rather than measurement artifacts.

In this workflow, we apply the following preprocessing techniques:

**Standard Normal Variate (SNV):** Corrects multiplicative scatter effects by centering and scaling each spectrum.

**Savitzky–Golay Derivative:** Removes baseline shifts and enhances spectral resolution by differentiating the signal.


####**2.3.1 Visualization of Raw Spectra** - examining unprocessed spectra before applying corrections which helps identify baseline drifts, outliers, and regions affected by scattering, guiding the choice of appropriate preprocessing methods.


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
plt.ylabel("Absorbance")
plt.tight_layout()
plt.show()

```


    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_11_0.png)
    



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

####**2.3.2 Visualization — After Preprocessing**


```python
plt.figure(figsize=(8,5))
for i in range(10):
    plt.plot(wavelengths, X_pre.iloc[i,:], color='grey', alpha=0.7)
plt.gca().invert_xaxis()
plt.title("Preprocessed Spectra (SNV + Savitzky–Golay 1st Derivative)")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Absorbance")
plt.tight_layout()
plt.show()

```


    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_14_0.png)
    


###**2.4 PCA Before and After Outlier Removal**

Principal Component Analysis (PCA) reduces high-dimensional spectral data into a smaller set of uncorrelated components (principal components). Each component captures a portion of the variance in the dataset. PCA is valuable for visualizing spectral structure and detecting potential outliers.

**Why Remove Outliers?**

Outliers may arise from measurement errors, mislabeling, or extreme soil compositions. Removing them improves model stability and ensures the predictive models learn from representative patterns.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

# -----------------------
# CONFIG
# -----------------------
ALPHA = 0.95        # 0.90 or 0.95
N_PCS = 3           # dimensions used for MD outlier detection
DOT_INLIER = dict(color='gray', alpha=0.7, s=20)
DOT_OUTLIER = dict(color='red',  alpha=0.8, s=25)
LINE_ELL = dict(edgecolor='black', linestyle='--', facecolor='none', lw=1.2)

# -----------------------
# Standardize and PCA
# -----------------------
X_scaled = StandardScaler().fit_transform(X_pre)   # X_pre = SNV + SG preprocessed spectra (n x p)
pca = PCA(n_components=N_PCS)
Z = pca.fit_transform(X_scaled)                    # PCA scores (n x N_PCS)
expl_var = pca.explained_variance_ratio_ * 100     # % explained per PC

# -----------------------
# 3D Mahalanobis for outliers (df = N_PCS)
# -----------------------
mu = Z.mean(axis=0)
S  = np.cov(Z.T)
Sinv = np.linalg.inv(S)
# squared MD
d2_3d = np.einsum('ij,jk,ik->i', Z - mu, Sinv, Z - mu)
thr_3d = chi2.ppf(ALPHA, df=N_PCS)                 # threshold in squared distance
outliers = d2_3d > thr_3d

n = len(Z)
n_out = int(outliers.sum())
pct_out = 100.0 * n_out / n
print(f"Total samples: {n}")
print(f"Outliers detected (ALPHA={ALPHA:.2f}, {N_PCS}D): {n_out} ({pct_out:.2f}%)")
print(f"Samples remaining after removal: {n - n_out}")

# -----------------------
# Helpers for 2D ellipse/coverage
# -----------------------
def plot_confidence_ellipse(ax, x, y, alpha=0.90, **kwargs):
    """
    Draws the alpha-level confidence ellipse in 2D assuming Gaussian scores.
    Uses chi-square(df=2) -> converts to n_std for ellipse axes.
    """
    XY = np.column_stack([x, y])
    C = np.cov(XY, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # convert alpha to 'std' scale for the ellipse axes
    n_std = np.sqrt(chi2.ppf(alpha, df=2))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=XY.mean(axis=0), width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ell)

def coverage_2d(x, y, alpha=0.90):
    """
    Returns boolean mask of points inside the 2D alpha-ellipse and coverage %
    """
    XY = np.column_stack([x, y])
    mu = XY.mean(axis=0)
    S  = np.cov(XY, rowvar=False)
    Sinv = np.linalg.inv(S)
    d2 = np.einsum('ij,jk,ik->i', XY - mu, Sinv, XY - mu)
    thr = chi2.ppf(alpha, df=2)
    inside = d2 <= thr
    return inside, inside.mean()*100.0

pairs = [(0, 1), (1, 2), (0, 2)]
alpha_lbl = f"{int(ALPHA*100)}%"

# -----------------------
# BEFORE (with outliers highlighted)
# -----------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
handles_for_legend = None

for i, (a, b) in enumerate(pairs):
    x, y = Z[:, a], Z[:, b]
    m1 = axs[i].scatter(x[~outliers], y[~outliers], label='Inliers', **DOT_INLIER)
    m2 = axs[i].scatter(x[outliers],  y[outliers],  label='Outliers', **DOT_OUTLIER)
    plot_confidence_ellipse(axs[i], x, y, alpha=ALPHA, **LINE_ELL)
    inside_mask, covg = coverage_2d(x, y, alpha=ALPHA)
    axs[i].text(0.02, 0.98, f"2D coverage ≈ {covg:.1f}%", transform=axs[i].transAxes,
                ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    axs[i].set_xlabel(f"PC{a+1} ({expl_var[a]:.1f}%)")
    axs[i].set_ylabel(f"PC{b+1} ({expl_var[b]:.1f}%)")
    # capture handles once (from first axis) for a shared legend
    if handles_for_legend is None:
        handles_for_legend = [m1, m2]

# remove per-axis legends and add one global legend
for ax in axs:
    ax.legend_.remove() if ax.legend_ is not None else None

# leave space on the right and place legend there
plt.subplots_adjust(right=0.84)
fig.legend(handles=handles_for_legend, labels=['Inliers', 'Outliers'],
           loc='center left', bbox_to_anchor=(0.86, 0.5), frameon=False)

fig.suptitle(f"Figure A: PCA with {int(ALPHA*100)}% Confidence Ellipse",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 0.84, 1])  # keep space for the legend
plt.show()

# -----------------------
# 4) compute 2D ellipse & show coverage
# -----------------------
Z_clean = Z[~outliers]
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i, (a, b) in enumerate(pairs):
    x, y = Z_clean[:, a], Z_clean[:, b]
    axs[i].scatter(x, y, **DOT_INLIER)
    plot_confidence_ellipse(axs[i], x, y, alpha=ALPHA, **LINE_ELL)
    inside_mask, covg = coverage_2d(x, y, alpha=ALPHA)
    axs[i].text(0.02, 0.98, f"2D coverage ≈ {covg:.1f}%", transform=axs[i].transAxes,
                ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    axs[i].set_xlabel(f"PC{a+1} ({expl_var[a]:.1f}%)")
    axs[i].set_ylabel(f"PC{b+1} ({expl_var[b]:.1f}%)")

plt.suptitle(f"Figure B: PCA After Outlier Removal ({alpha_lbl} Confidence Ellipse)", fontsize=13)
plt.tight_layout()
plt.show()

# -----------------------
# Clean dataset for modeling
# -----------------------
if hasattr(X_pre, "loc"):
    X_clean = X_pre.loc[~outliers]
else:
    X_clean = X_pre[~outliers]

df_clean = df.loc[~outliers].reset_index(drop=True)

```

    Total samples: 517
    Outliers detected (ALPHA=0.95, 3D): 38 (7.35%)
    Samples remaining after removal: 479



    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_16_1.png)
    



    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_16_2.png)
    


**Figure A — “PCA with 90% Confidence Ellipse — Outlier Detection”**

*   Each dot = one soil sample summarized by PCA (PC1 vs PC2, PC2 vs PC3, PC1 vs PC3).
*   Grey points = samples that look typical relative to the rest.
*   Red points = samples flagged as outliers.
*   The dashed ellipse encloses about 90% of the most typical samples (the main “cloud”).

If a point lies inside the ellipse → its PCA scores are consistent with the group.

If a point lies outside → it’s unusually far from the cluster after preprocessing, often due to measurement issues (spikes, poor contact, mislabel, abnormal baseline) rather than true soil chemistry.

Patterns across the three panels tell you whether the unusualness is along PC1, PC2, or PC3. Outliers can destabilize models (PLSR/MLP), inflate error, and distort interpretations. We remove a small number of clear outliers to keep the calibration set clean and reliable.

**Figure B — “PCA After Outlier Removal (90% Confidence Ellipse)”**

Only cleaned samples remain (grey). The ellipse still marks the 95% region, but now the cloud is tighter and more elliptical—exactly what we want. The data now show a coherent structure with fewer extreme points. This typically leads to more stable model fitting and better generalization on test data.

**Why do some points remain outside the ellipse?**

The dashed ellipse shows the 90% confidence region of the PCA scores after cleaning. By definition, ~10% of typical samples can fall outside. Also, outliers were detected in 3D PC space, while the plots are 2D projections; a point can be inside the 3D boundary but outside a 2D ellipse. This is normal. We only remove samples that look like measurement artefacts on their spectra; the rest are valid and help the model learn real soil variability.

---
##**3 Modeling and Evaluation**

Two regression approaches are compared: PLSR and MLP

###**3.1 PLSR (Partial Least Squares Regression)**

PLSR projects both predictors and responses to a shared latent space, ideal for multicollinear spectral data.




```python
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def rmse(y, yhat):
    """Root Mean Square Error"""
    return float(np.sqrt(mean_squared_error(y, yhat)))

def bias(y, yhat):
    """Average bias"""
    return float(np.mean(yhat - y))

# ----------------------------------------------------
# Data setup (X_pre or X_clean already in memory)
# ----------------------------------------------------
target_col = "eoc_tot_c"
nice_name  = "Organic Carbon (%)"

# Ensure X and y are aligned (handle NaN safely)
y = df[target_col]
valid_rows = ~y.isna()   # Boolean mask of rows where target exists
X_sel = X_pre.loc[valid_rows].copy() if hasattr(X_pre, "loc") else X_pre[valid_rows.values]
y_sel = y[valid_rows].values

# Split into training and testing sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_sel.values, y_sel, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
```

    Training samples: 361, Testing samples: 156



```python
# ----------------------------------------------------
# PLSR model with component tuning
# ----------------------------------------------------

max_k = min(25, X_train.shape[1])
cv_r2_scores = []

for k in range(2, max_k + 1):
    pls = PLSRegression(n_components=k)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2 = []
    for tr, va in kf.split(X_train):
        pls.fit(X_train[tr], y_train[tr])
        yv = pls.predict(X_train[va]).ravel()
        fold_r2.append(r2_score(y_train[va], yv))
    cv_r2_scores.append(np.mean(fold_r2))

best_k = np.argmax(cv_r2_scores) + 2  # offset because range starts at 2

plt.figure(figsize=(6,4))
plt.plot(range(2, max_k+1), cv_r2_scores, 'o-', lw=2, color='gray')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.xlabel("Number of PLS Components")
plt.ylabel("Mean CV R²")
plt.title("Component Tuning — Organic Carbon (%)")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Fit final model
# ------------------------------------------------------------------
pls = PLSRegression(n_components=best_k)
pls.fit(X_train, y_train)
y_pred_plsr = pls.predict(X_test).ravel()

R2_plsr = r2_score(y_test, y_pred_plsr)
RMSE_plsr = rmse(y_test, y_pred_plsr)
Bias_plsr = bias(y_test, y_pred_plsr)

print(f"✅ PLSR Best k={best_k}, R²={R2_plsr:.3f}, RMSE={RMSE_plsr:.3f}, Bias={Bias_plsr:.3f}")

# ------------------------------------------------------------------
# Regression Coefficients vs Wavelength
# ------------------------------------------------------------------
wavelengths = np.array(X_pre.columns, dtype=float) if hasattr(X_pre, "columns") else np.arange(X_train.shape[1])
coefs = pls.coef_.ravel()

plt.figure(figsize=(8,4))
plt.plot(wavelengths, coefs, color='gray', lw=1.5)
plt.title(f"PLSR Coefficients — {best_k} Components")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Regression Coefficient")
plt.gca().invert_xaxis()  # MIR convention: high → low wavenumber
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"✅ PLSR Best components = {best_k}, R²={R2_plsr:.3f}, RMSE={RMSE_plsr:.3f}, Bias={Bias_plsr:.3f}")



```


    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_20_0.png)
    


    ✅ PLSR Best k=19, R²=0.939, RMSE=0.465, Bias=-0.026



    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_20_2.png)
    


    ✅ PLSR Best components = 19, R²=0.939, RMSE=0.465, Bias=-0.026


The first plot shows how the performance of the PLSR model changes as we increase the number of components. Each point represents the mean cross-validated R² for a given number of components. At first, R² rises quickly as the model captures more useful information from the spectra. After around 19 components, the improvement levels off, indicating that adding more components no longer helps and may start fitting noise. Therefore, 19 components provide the best balance between accuracy and simplicity for predicting organic carbon.

The second plot displays the PLSR regression coefficients for these 19 components across the MIR wavenumber range. The peaks and dips show which spectral regions contribute most strongly to the prediction of organic carbon. Positive peaks indicate wavelengths that increase predicted values, while negative ones decrease them. These active regions often correspond to molecular vibrations of carbon-containing functional groups, revealing the spectral features most related to organic matter in the soil. The taller the peak or dip (in absolute value), the more influential that wavelength is in the model.

###**3.2 MLP (Multilayer Perceptron):**

A feed-forward neural network that can learn nonlinear relationships.


```python
# ----------------------------------------------------
# MLP Neural Network
# ----------------------------------------------------

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Build a compact, robust MLP
inputs = keras.Input(shape=(X_train.shape[1],))
x = layers.Dense(128, activation="relu")(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1)(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# Train
es = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=8,
    callbacks=[es],
    verbose=0
)


```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1740</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">222,848</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">8,256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">65</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">231,169</span> (903.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">231,169</span> (903.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
y_pred_mlp = model.predict(X_test).ravel()
R2_mlp = r2_score(y_test, y_pred_mlp)
RMSE_mlp = rmse(y_test, y_pred_mlp)
Bias_mlp = bias(y_test, y_pred_mlp)

print(f"✅ MLP R²={R2_mlp:.3f}, RMSE={RMSE_mlp:.3f}, Bias={Bias_mlp:.3f}")

```

    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step 
    ✅ MLP R²=0.955, RMSE=0.398, Bias=0.037



```python
import matplotlib.pyplot as plt
import numpy as np

# Extract losses
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Identify best epoch (lowest validation loss)
best_epoch = np.argmin(val_loss)
best_val = val_loss[best_epoch]
best_train = train_loss[best_epoch]

# Plot
plt.figure(figsize=(6,4))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.axvline(best_epoch, color='red', linestyle='--', label=f"Best Epoch = {best_epoch+1}")
plt.scatter(best_epoch, best_val, color='red', alpha = 0.7, s=50, zorder=5)

plt.title("MLP Training Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Best epoch: {best_epoch+1}")
print(f"Train Loss = {best_train:.4f}")
print(f"Val Loss   = {best_val:.4f}")

```


    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_25_0.png)
    


    Best epoch: 80
    Train Loss = 0.1671
    Val Loss   = 0.2569


The model learned steadily over time, reached its best validation accuracy around epoch 80, and stopped before it began overfitting

###**3.3 Scatterplots**


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------
# Compute metrics for both models
# ---------------------------------------------------------------------
metrics = {
    "Model": ["PLSR", "MLP"],
    "R²": [
        r2_score(y_test, y_pred_plsr),
        r2_score(y_test, y_pred_mlp)
    ],
    "RMSE": [
        rmse(y_test, y_pred_plsr),
        rmse(y_test, y_pred_mlp)
    ],
    "Bias": [
        bias(y_test, y_pred_plsr),
        bias(y_test, y_pred_mlp)
    ]
}

results_df = pd.DataFrame(metrics)
print("✅ Performance Summary:")
display(results_df.round(3))

# ---------------------------------------------------------------------
# Scatterplots
# ---------------------------------------------------------------------
plt.figure(figsize=(10,4))

# Scatter for PLSR
plt.subplot(1,2,1)
sns.scatterplot(x=y_test, y=y_pred_plsr, s=25, color='gray', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
plt.title(f"PLSR \nR²={results_df.loc[0,'R²']:.2f}, RMSE={results_df.loc[0,'RMSE']:.2f}, Bias={results_df.loc[0,'Bias']:.2f}")
plt.xlabel("Observed OC (%)")
plt.ylabel("Predicted OC (%)")
plt.grid(alpha=0.3)

# Scatter for MLP
plt.subplot(1,2,2)
sns.scatterplot(x=y_test, y=y_pred_mlp, s=25, color='steelblue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
plt.title(f"MLP \nR²={results_df.loc[1,'R²']:.2f}, RMSE={results_df.loc[1,'RMSE']:.2f}, Bias={results_df.loc[1,'Bias']:.2f}")
plt.xlabel("Observed OC (%)")
plt.ylabel("Predicted OC (%)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

```

    ✅ Performance Summary:




  <div id="df-bb283efb-a1a3-486c-a68f-7561f96291e9" class="colab-df-container">
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
      <th>Model</th>
      <th>R²</th>
      <th>RMSE</th>
      <th>Bias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PLSR</td>
      <td>0.939</td>
      <td>0.465</td>
      <td>-0.026</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MLP</td>
      <td>0.955</td>
      <td>0.398</td>
      <td>0.037</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bb283efb-a1a3-486c-a68f-7561f96291e9')"
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
        document.querySelector('#df-bb283efb-a1a3-486c-a68f-7561f96291e9 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bb283efb-a1a3-486c-a68f-7561f96291e9');
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


    <div id="df-64d05347-53ed-4a1b-935f-1c82fad49397">
      <button class="colab-df-quickchart" onclick="quickchart('df-64d05347-53ed-4a1b-935f-1c82fad49397')"
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
            document.querySelector('#df-64d05347-53ed-4a1b-935f-1c82fad49397 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




    
![png](hands_on_exercise_instructor_files/hands_on_exercise_instructor_28_2.png)
    


###**3.4 Model Comparison**

The figure above compares the predictive performance of the PLSR and the MLP models for estimating soil organic carbon (OC%) from MIR spectra. Each point represents a soil sample, with the observed OC values on the x-axis and the model-predicted values on the y-axis. The red dashed line is the 1:1 reference line, where perfect predictions would fall.

Both models show strong agreement between observed and predicted values, indicating high accuracy. The PLSR model achieved an R² of 0.94 and an RMSE of 0.47, suggesting it captures most of the variation in organic carbon using linear relationships between spectral features and OC content. The MLP model performed slightly better, with an R² of 0.95 and an RMSE of 0.40, reflecting its ability to learn nonlinear patterns in the data. The small bias values (–0.03 for PLSR and +0.04 for MLP) indicate that both models have minimal systematic error, with PLSR slightly underestimating and MLP slightly overestimating organic carbon.

Overall, both approaches produced reliable predictions, but the MLP demonstrated a modest improvement in precision and generalization compared to PLSR, especially at higher carbon concentrations.

---
##**4 References**

1. Barra, Issam, Stephan M. Haefele, Ruben Sakrabani, and Fassil Kebede. 2021. “Soil Spectroscopy with the Use of Chemometrics, Machine Learning and Pre-Processing Techniques in Soil Diagnosis: Recent Advancesa Review.” TrAC Trends in Analytical Chemistry 135 (February): 116166.

2. Chang, Cheng-Wen, David Laird, Maurice J Mausbach, and Charles R Hurburgh Jr. 2001. “Near-Infrared Reflectance Spectroscopy–Principal Components Regression Analyses of Soil Properties.” Soil Science Society of America Journal 65 (2): 480. https://doi.org/10.2136/sssaj2001.652480x.

3. Seybold, Cathy A., Rich Ferguson, Doug Wysocki, Scarlett Bailey, Joe Anderson, Brian Nester, Phil Schoeneberger, et al. 2019. “Application of Mid-Infrared Spectroscopy in Soil Survey.” Soil Science Society of America Journal 83 (6): 1746–59. https://doi.org/10.2136/sssaj2019.06.0205.



```python
!jupyter nbconvert --to markdown "/content/drive/My Drive/Colab Notebooks/hands_on_exercise_instructor.ipynb"

```

    [NbConvertApp] Converting notebook /content/drive/My Drive/Colab Notebooks/hands_on_exercise_instructor.ipynb to markdown
    [NbConvertApp] Support files will be in hands_on_exercise_instructor_files/
    [NbConvertApp] Writing 58626 bytes to /content/drive/My Drive/Colab Notebooks/hands_on_exercise_instructor.md

