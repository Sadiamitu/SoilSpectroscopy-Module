# Introduction to Soil Spectral Modeling

**Author:** Sadia Mitu  
**University of Nebraska–Lincoln**  

---

## 📘 Overview
This teaching module introduces graduate students to the principles and applications of **machine learning and AI for soil spectroscopy analysis**. The session blends theoretical understanding of soil spectral sensing with practical, hands-on modeling using **Python**.

The materials are designed for an audience with **diverse academic backgrounds** — including soil science, agronomy, data science, and engineering. The goal is to help all participants develop a working understanding of how soil reflectance spectra can be analyzed using modern AI tools.

---

## 🎯 Learning Objectives
By the end of this module, students will be able to:

| Bloom Level | Objective |
|--------------|------------|
| **Understand** | Explain the physical basis and interpretation of soil spectra (VIS–NIR–MIR). |
| **Apply** | Preprocess spectra using SNV and Savitzky–Golay filters. |
| **Analyze** | Build and evaluate PLSR and deep learning models to predict soil properties. |
| **Evaluate** | Compare performance (R², RMSE, RPD) between linear and nonlinear models. |
| **Create** | Design improved soil spectral modeling workflows. |

---

## 🧩 Included Materials

| File | Description |
|------|--------------|
| `Lecture1-basic_reduced.pptx` | 30-min lecture covering spectroscopy principles and ML overview. |
| `MIR_spectra.csv` | Subset of MIR soil spectra (500+ samples; USDA-KSSL). |
| `hands_on_exercise_instructor.ipynb` | Full instructor notebook with solutions and commentary. |
| `hands_on_exercise_student.ipynb` | Guided student notebook with TODO sections for practice. |
| `hands_on_exercise_instructor.html` / `.md` | Rendered versions for quick viewing or offline reference. |
| `Instructor_Guide.pdf` | (Optional) Teaching notes: flow, timing, key discussion prompts. |

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
1. Why might deep learning outperform PLSR for certain soil properties?
2. How can calibration transfer improve model performance across spectrometers?
3. What preprocessing strategies could help in low-data or noisy conditions?

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

---

## 🧭 Suggested Directory Structure
```
AI_in_Soil_Spectroscopy_Teaching_Module/
├── Lecture1-basic_reduced.pptx
├── MIR_spectra.csv
├── hands_on_exercise_instructor.ipynb
├── hands_on_exercise_student.ipynb
├── hands_on_exercise_instructor.html
├── hands_on_exercise_instructor.md
├── Instructor_Guide.pdf
└── README.md
```

---

**Contact:** smitu2@huskers.unl.edu  
**License:** Educational use only (UNL teaching module, 2025)
