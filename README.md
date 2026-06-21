# Big-Mart-Sale-prediction-ML
# 🛒 Big Mart Sales Prediction using XGBoost Regression

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![Status](https://img.shields.io/badge/Project-Completed-success)


## 📌 Project Overview

This project predicts **Big Mart Item Outlet Sales** using a Machine Learning regression model.

The complete workflow:

```
Data Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Encoding
        ↓
Train/Test Split
        ↓
XGBoost Regression
        ↓
Model Evaluation
        ↓
Sales Prediction
```


## 📂 Dataset

Dataset:

**Big Mart Sales Dataset**

Features:

- Item Identifier
- Item Weight
- Item Fat Content
- Item Visibility
- Item Type
- Item MRP
- Outlet Identifier
- Outlet Establishment Year
- Outlet Size
- Outlet Location Type
- Outlet Type


Target:

```
Item_Outlet_Sales
```


Dataset Information:

```
Rows: 8523
Columns: 12
```


---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost


---

# ⚙️ Data Preprocessing


## Missing Value Handling


### Item Weight

Missing values replaced using mean value.


### Outlet Size

Missing values replaced using mode based on Outlet Type.



---

# 📊 Exploratory Data Analysis


## Item Weight Distribution

![Item Weight](count%20vs%20item%20weight.png)


## Item Visibility Distribution

![Item Visibility](count%20vs%20item%20visibility.png)


## Item MRP Distribution

![Item MRP](count%20vs%20item%20MRP.png)


## Item Fat Content

![Item Fat](count%20vs%20item%20fat%20content.png)


## Item Type Analysis

![Item Type](count%20vs%20item%20type.png)


## Outlet Size Analysis

![Outlet Size](count%20vs%20item%20outlet%20size.png)


## Outlet Establishment Year

![Outlet Year](count%20vs%20outlet%20establishment%20year.png)


## Item Outlet Sales

![Sales](count%20vs%20item%20outlet%20sale.png)



---

# 🔄 Feature Encoding


Categorical variables converted into numerical values using:

```
Label Encoding
```


Encoded Features:

- Item Identifier
- Item Fat Content
- Item Type
- Outlet Identifier
- Outlet Size
- Outlet Location Type
- Outlet Type


---

# 🤖 Machine Learning Model


## XGBoost Regression


Model:

```python
XGBRegressor()
```


Training:

```python
model.fit(X_train,Y_train)
```


---

# 📈 Model Evaluation


Train-Test Split:

```
Training Data : 80%

Testing Data : 20%
```


Evaluation Metric:

## R² Score


Results:


```
Training R² Score : 0.87

Testing R² Score  : 0.50
```


---

# 🔮 Prediction System


The trained model predicts future sales based on item and outlet information.


Flow:

```
Input Data

      ↓

Data Processing

      ↓

XGBoost Model

      ↓

Predicted Item Outlet Sales
```


---

# 📁 Project Structure


```
Big-Mart-Sales-Prediction

│
├── Train.csv
│
├── big mart sale prediction.py
│
├── README.md
│
├── count vs item MRP.png
├── count vs item fat content.png
├── count vs item outlet sale.png
├── count vs item type.png
├── count vs item visibility.png
├── count vs item weight.png
├── count vs outlet establishment year.png
├── count vs outlet size.png
│
└── requirements.txt

```


---

# ▶️ How To Run


Clone repository:


```bash
git clone YOUR_GITHUB_LINK
```


Install dependencies:


```bash
pip install -r requirements.txt
```


Run project:


```bash
python "big mart sale prediction.py"
```


---

# 📦 Requirements


```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```


---

# 👨‍💻 Author

Allen Christian | Patent holder


---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub.
