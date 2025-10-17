# Introduction to Soil Spectral Modeling

**Author:** Sadia Mitu  
**University of Nebraska–Lincoln**  

---

## 📘 Overview
This teaching module introduces graduate students to the fundamentals of soil spectroscopy and its integration with machine learning techniques for soil property prediction. The goal is to develop a clear understanding of how soil spectral data are structured, how Partial Least Squares Regression (PLSR) and Multilayer Perceptron (MLP) models operate, and how these methods can be used to model and interpret soil properties from spectral measurements.

Through step-by-step exercises, students will preprocess mid-infrared (MIR) spectra, apply dimensionality reduction, build PLSR and MLP models, and compare their predictive performance using quantitative metrics.

---

## 🎯 Learning Objectives
By the end of this module, students will be able to:

- Explain the fundamental principles of soil spectroscopy and how spectral data relate to soil properties.

- Apply preprocessing techniques such as Standard Normal Variate (SNV) and Savitzky–Golay filtering to improve spectral quality.

- Develop and evaluate predictive models using Partial Least Squares Regression (PLSR) and Multilayer Perceptron (MLP) approaches.

- Compare the performance of linear and nonlinear models using metrics such as R², RMSE, and Bias.

- Interpret model outputs to understand the relationship between spectral features and soil attributes.

---

## 🧩 Included Materials

| File | Description |
|------|--------------|
| `Lecture-Introduction to Soil Spectral Modeling.pptx` | 30-min lecture covering spectroscopy principles and ML overview. |
| `MIR_spectra.csv` | Subset of MIR soil spectra (500+ samples; USDA-KSSL). |
| `hands_on_exercise_instructor.ipynb` | Full instructor notebook with solutions and commentary. |
| `hands_on_exercise_student.ipynb` | Guided student notebook with TODO sections for practice. |
| `hands_on_exercise_instructor.html` / `.md` | Rendered versions for quick viewing or offline reference. |

---

## 🧠 Exercise Summary
The practical activity walks students through a complete soil spectral modeling workflow:

1. **Data loading & exploration** (MIR spectra of 599–3953 cm⁻¹)
2. **Spectral preprocessing:** Standard Normal Variate (SNV) & Savitzky–Golay filtering
3. **Exploratory PCA** to detect structure and outliers
4. **PLSR modeling** — baseline linear regression approach
5. **MLP deep learning** — nonlinear modeling and performance comparison
6. **Metrics & visualization:** R², RMSE, scatter plots
7. **Discussion:** interpretability, generalization, and calibration transfer

---

## 🧰 Requirements
To run the notebooks:
```bash
pip install pandas numpy scikit-learn matplotlib scipy tensorflow
```
Or in Google Colab:
```python
!pip install pandas numpy scikit-learn matplotlib scipy tensorflow
```

---

## 💡 Homework / Reflection Questions
1. Why might a Multilayer Perceptron (MLP) outperform PLSR for certain soil properties?
2. How does spectral preprocessing (e.g., SNV, Savitzky–Golay) influence model accuracy and stability?
3. What are the main trade-offs between interpretability and predictive performance when comparing PLSR and MLP models?

---

## 📈 Expected Learning Outcomes
Students completing the exercise will:
- Demonstrate ability to preprocess spectral data and build regression models.
- Quantitatively evaluate model accuracy and generalization.
- Interpret the link between spectral features and soil properties.
- Appreciate the balance between model complexity, interpretability, and robustness.

---

## 📚 References
- Viscarra Rossel, R. A. et al. (2006). *Visible, near infrared, mid infrared or combined diffuse reflectance spectroscopy for simultaneous assessment of various soil properties.* **Geoderma, 131**, 59–75.
- Mitu, S. et al. (2024). *Deep learning and calibration transfer for soil spectroscopy.* (In preparation).


