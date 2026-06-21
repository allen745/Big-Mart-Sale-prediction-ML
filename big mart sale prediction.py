# data --> data pre processing --> data analysis --> train test data --> XG boost regression --> evalution

# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn import metrics

# data collection
# loading data to csv file to pandas dataframe
big_mart_data = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\Big mart\Train.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(big_mart_data.head())

# number of datapoints and number of features
print(big_mart_data.shape)
# getting some informational data
print(big_mart_data.info())

# categorical features
# - Item_Identifier
# - Item_Fat_Content
# - Item_Type
# - Outlet_Identifier
# - Outlet_Size
# - Outlet_Location_Type
# - Outlet_Type
# checkin null values
print(big_mart_data.isnull().sum())

# handling null values
# mean --> avg values
# mode --> most repeated values

# mean value of Item_Weight
big_mart_data['Item_Weight'].mean()
# filling the null values in Item_Weight columns with mean values
big_mart_data['Item_Weight'] = big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean())
print(big_mart_data.isnull().sum())

# replacing the values in Outlet_Size
mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode().iloc[0]))
print(mode_of_outlet_size)

nan_values = big_mart_data['Outlet_Size'].isnull()
print(nan_values.sum())

big_mart_data.loc[nan_values, 'Outlet_Size'] = big_mart_data.loc[nan_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size[x].values[0])
print(big_mart_data.isnull().sum())


# data analysis
# statical measures about the data
print(big_mart_data.describe())

# numerical features

sns.set()

# Item_Weight distribution
plt.figure(figsize=(10,10))
sns.histplot(big_mart_data['Item_Weight'], kde=True)
plt.show()

# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_Visibility'], kde=True)
plt.show()

 # Item MRP distribution
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_MRP'], kde=True)
plt.show()

# Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_Outlet_Sales'], kde=True)
plt.show()

# Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()

# Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

# Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

# Outlet_Size column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.show()

big_mart_data['Item_Fat_Content'].value_counts()
big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
big_mart_data['Item_Fat_Content'].value_counts()

# label encoder
encoder = LabelEncoder()

big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

print(big_mart_data.head())

X = big_mart_data.drop(columns='Item_Outlet_Sales')
Y = big_mart_data['Item_Outlet_Sales']

# Splitting the data into Training data & Testing Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Machine Learning Model Training
# XGBoost Regressor

regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# evalution
# prediction on training data
training_data_prediction = regressor.predict(X_train)
# r square value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print("R square value: ", r2_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print("R square value: ", r2_test)
