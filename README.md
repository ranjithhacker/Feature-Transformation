# Ex-06-Feature-Transformation
AIM

To read the given data and perform Feature Transformation process and save the data to a file.

EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM

STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Transformation techniques to all the features of the data set

STEP 4 Save the data to the file CODE

Program :

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

df=pd.read_csv("/content/Data_to_Transform.csv")

print(df)

df.head()

![image](https://user-images.githubusercontent.com/113016781/197590714-952215d7-4f53-4432-8fe1-ec5e8c58a225.png)

df.isnull().sum()

![image](https://user-images.githubusercontent.com/113016781/197590843-b2601bc3-d94d-45a5-932a-7d97c5e62047.png)

df.info()

df.describe()

![image](https://user-images.githubusercontent.com/113016781/197590975-5d6660ba-5fe5-4a2a-ac2f-63d5750037dd.png)

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591090-b6157646-46d2-4f85-92da-2d462e42cce6.png)

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591205-1c1d0dbe-59f9-45db-9ea8-f376abd3692f.png)

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591293-524fce0d-9484-40b6-bcc9-5492db40a435.png)

df4=df.copy()

df4['ModerateNegativeSkew_1'],parameters=stats.yeojohnson(df4.ModerateNegativeSkew)

![image](https://user-images.githubusercontent.com/113016781/197591376-f760c1da-08cc-4099-aacd-b122ca8c1dd1.png)

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591450-6b498d16-f8af-44c4-9156-231f8d48fdce.png)

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)

plt.show()

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591576-bd615579-aa11-47ba-ba13-4577da65debb.png)


df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591674-b52881f6-fb1e-4460-9985-f5d8da9dcc5d.png)

df3 = df.copy()

df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591761-bdeb86df-ba70-4db5-be65-b7e4e16e893a.png)


df4 = df.copy()

df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)

sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591894-fed717d3-b10f-4e1d-83e2-e7c0f58fcfdc.png)


from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df4['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df4[['ModerateNegativeSkew']]))

sm.qqplot(df4['ModerateNegativeSkew_2'],fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/113016781/197591967-be04707a-32a6-4913-8ad4-4250f013f496.png)

RESULT:

Thus the Feature Transformation for the given datasets had been executed successfully
