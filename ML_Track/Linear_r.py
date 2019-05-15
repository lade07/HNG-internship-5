# Linear Regression with Python to predict and decide whether an Ecommerce company based in New York City that sells clothing online and also in-store style and clothing advice sessions should focus their efforts on their mobile app experience or their website.

import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd 
import seaborn as sns
%matplotlib inline
from scipy import stats
import numpy as np 
from dfply import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# data import
Ecom = pd.read_csv("https://raw.githubusercontent.com/lade07/myOpenSet/master/Ecommerce%20Customers")
Ecom.head()
Ecom.info()
Ecom.describe()
Ecom.columns


sns.kdeplot(Ecom['Avg. Session Length'])
sns.kdeplot(Ecom['Yearly Amount Spent'])
# check relationship between time on app and yearly amount spent
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=Ecom)

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=Ecom, kind = 'kde')

# check relationship between time on website and yearly amount spent
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=Ecom)

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=Ecom, kind='kde')

# check relationship between time on application and length of membership
sns.jointplot(x='Time on App',y='Length of Membership',data=Ecom, kind='hex')

# complete data exploratory using pairplot to understand data better and which variable has the most correlation with the metric measured
sns.pairplot(Ecom)

# regression plot showing fit line
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=Ecom) # fitted line shows a good fit

# train and test datasets

# linear model instance
lm = LinearRegression()

# assign x and y
x = Ecom[['Avg. Session Length', 'Time on App',
'Time on Website', 'Length of Membership']]

y = Ecom['Yearly Amount Spent']

# data split using train_test_split func
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=101)

# model fit
lm.fit(x_train,y_train)

## Model Evaluation
### Let's evaluate the model by checking out it's coefficients and how we can interpret them.
print(lm.intercept_)
lm.coef_
coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff_df 


# Prediction
predictions = lm.predict(x_test)
predictions
y_test

# scatter plot of predictions and actual values
plt.scatter(y_test,predictions)

# distribution of residuals
sns.distplot((y_test-predictions)) # normal distribution indicates right choice of data

# metrics evaluation
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


