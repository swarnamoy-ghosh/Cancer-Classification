# Breast Cancer Image Classification (IDC vs Non-IDC)

This project involves classifying histopathological image patches of breast cancer tissue to detect **Invasive Ductal Carcinoma (IDC)** using **Convolutional Neural Networks (CNNs)**. The dataset contains image patches extracted from whole slide images scanned at 40x magnification.

---

## 📁 Dataset Overview

- **Source**: 162 whole mount slide images of breast cancer specimens  
- **Total Patches**: 277,524 images (50x50 pixels)  
- **Classes**:
  - **Class 0**: Non-IDC
  - **Class 1**: IDC-positive

- **File naming format**:  
  `u_xX_yY_classC.png`  
  - `u`: Patient ID  
  - `X`, `Y`: Patch coordinates  
  - `C`: Class label (0 or 1)

---

## 🧠 Objective

To build a robust deep learning model capable of classifying small histopathology image patches into IDC-positive or non-IDC classes for aiding early cancer diagnosis.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Pillow (PIL) for image handling
- Scikit-learn for metrics

---

## 🔍 Project Workflow

1. **Data Preprocessing**
   - Loaded and resized images
   - Converted to arrays and normalized pixel values
   - Labeled data extracted from filenames
   - Balanced dataset using data augmentation (if applied)

2. **Model Building**
   - CNN architecture with multiple convolutional and pooling layers
   - Dropout and BatchNormalization used
   - Binary classification output (sigmoid activation)

3. **Training**
   - Trained with cross-entropy loss
   - Early stopping and model checkpointing used

4. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix and ROC-AUC curve plotted

---

## 📈 Results

- **Accuracy**: ~92% on test set
- **ROC-AUC**: ~0.94
- Used class weighting and data augmentation to handle imbalance
- Improved generalization with dropout and early stopping

---

## 📂 Folder Structure

```text
├── images/                # Folder containing all patch images
├── notebook.ipynb         # Jupyter notebook with full code
├── model/                 # Saved model and weights
├── results/               # Plots and evaluation metrics
└── README.md              # Project documentation
