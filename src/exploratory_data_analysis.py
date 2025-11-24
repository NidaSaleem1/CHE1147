"""Exploratory Data Analysis

EDA will only be done on the training set to prevent data leakage
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

trainDFLoc = os.path.join("data", "df_train.csv")
cleanedDFLoc = os.path.join("data", "cleaned_df.csv")

DF = pd.read_csv(cleanedDFLoc)
DF_TRAIN = pd.read_csv(trainDFLoc)

#Checking correlations to target variable
DF_TRAIN.corr()['CO2 Capacity (mmol/g)'].sort_values(ascending=False)

#Plotting Pie Chart to see distribution of support morphology

#Defining train indexes in original DF and name threshold
support_counts = DF.loc[DF_TRAIN.index]['CO2 Concentration (vol%)'].value_counts()
threshold = 1  # show labels only for slices > 6%

# Plot pie chart
plt.figure(figsize=(6,6))
wedges, texts, autotexts = plt.pie(
    support_counts,
    labels=None,
    autopct=lambda pct: f"{pct:.1f}%" if pct > threshold else '',  # shows percentage only if > threshold
    startangle=90,
    colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'],
    wedgeprops={'edgecolor':'black', 'linewidth':1}
)

for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2
    if (support_counts.iloc[i] / support_counts.sum() * 100) > threshold:
        x = 1.19 * np.cos(np.deg2rad(angle))
        y = 1.1 * np.sin(np.deg2rad(angle))
        plt.text(x, y, support_counts.index[i].astype(str)+'%', ha='center', va='center')

plt.title('CO2 Concentration (Vol-%) Pie Chart')
plt.show()

#Plotting Pie Chart to see distribution of Amine_structure

#Defining train indexes in original DF and name threshold
amine_counts = DF.loc[DF_TRAIN.index]['Amine_structure'].value_counts()

# Plot pie chart
plt.figure(figsize=(6,6))
wedges, texts, autotexts = plt.pie(
    amine_counts,
    labels=None,
    autopct=lambda pct: f"{pct:.1f}%" if pct > threshold else '',  # shows percentage only if > threshold
    startangle=90,
    colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'],
    wedgeprops={'edgecolor':'black', 'linewidth':1}
)

for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2
    if (amine_counts[i] / amine_counts.sum() * 100) > threshold:
        x = 1.2 * np.cos(np.deg2rad(angle))
        y = 1 * np.sin(np.deg2rad(angle))
        plt.text(x, y, amine_counts.index[i], ha='center', va='center')

plt.title('Amine Structure Pie Chart')
plt.show()

#Plotting target value histogram to see its distribution
sns.histplot(DF_TRAIN['CO2 Capacity (mmol/g)'], bins=12, kde=True, color='skyblue', edgecolor='black')
plt.title('Histogram - CO2 Capacity (mmol/g)')
plt.xlabel('CO2 Capacity (mmol/g)')
plt.ylabel('Frequency')

plt.show()

#Plotting target value QQplot to confirm its distribution
stats.probplot(DF_TRAIN['CO2 Capacity (mmol/g)'], dist='norm', plot=plt)
plt.title('QQ-Plot of CO2 Capacity (mmol/g)')
plt.show()

#Defining a function to calculate VIF
def find_vif(data):
        vif_df = pd.DataFrame(columns=['Variable', 'VIF'])
        x_vars = data.drop(columns=['CO2 Capacity (mmol/g)'])
        y_vars = x_vars.dropna()

        for i in np.arange(0, x_vars.shape[1]):
            linear_model = LinearRegression()
            vif_current = 0

            for j in np.arange(0, x_vars.shape[1]):
              if i==j:
                vif_current=vif_current
              else:
                linear_model.fit(x_vars.iloc[:,[i]], x_vars.iloc[:,j])
                pred_y = linear_model.predict(x_vars.iloc[:,[i]])
                vif_current = vif_current+1/((1 - r2_score(x_vars.iloc[:,i], pred_y)))
            vif_df.loc[i] = [x_vars.columns[i], vif_current]
        return vif_df

#Checking multicolinearity of features through VIF
vif_df = find_vif(DF_TRAIN)
vif_df.sort_values(by='VIF', ascending=False)