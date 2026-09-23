# MAGIC Gamma Telescope Classification with Logistic Regression

An end-to-end Machine Learning pipeline for binary classification on the **MAGIC Gamma Telescope Dataset** using **Logistic Regression**.

---

## 📌 Overview

This project classifies high-energy gamma particles versus background hadron noise. The workflow involves:
* Converting string labels into binary ground truth values.
* Splitting data into Training (60%), Validation (20%), and Testing (20%) sets.
* Standardizing numerical features using `StandardScaler`.
* Handling class imbalance on the training dataset via `RandomOverSampler`.
* Training and evaluating a **Logistic Regression** model.

---

## 📊 Dataset Overview

The dataset used is the **MAGIC Gamma Telescope Data Set** (`magic04.data`).

* **Target Variable (`class`)**:
  * `g` (Gamma) → Converted to `1`
  * `h` (Hadron) → Converted to `0`

* **Features**:
  * `fLength`: Major axis of ellipse [mm]
  * `fWidth`: Minor axis of ellipse [mm]
  * `fSize`: Log10 of total light signal
  * `fConc`: Ratio of light in 2 highest pixels
  * `fConc1`: Ratio of light in highest pixel
  * `fAsym`: Distance from highest pixel to center
  * `fM3Long`: 3rd root of 3rd moment along major axis
  * `fM3Trans`: 3rd root of 3rd moment along minor axis
  * `fAlpha`: Angle of major axis with vector to center [deg]
  * `fDist`: Distance from center to shower centroid [mm]

---

## 🛠️ Prerequisites & Installation

Install the required dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn imbalanced-learn
