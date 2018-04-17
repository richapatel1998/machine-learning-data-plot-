
In [11]:

import pandas as pd
%matplotlib inline
In [12]:

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv')
data.head()
Out[12]:
Unnamed: 0	TV	radio	newspaper	sales
0	1	230.1	37.8	69.2	22.1
1	2	44.5	39.3	45.1	10.4
2	3	17.2	45.9	69.3	9.3
3	4	151.5	41.3	58.5	18.5
4	5	180.8	10.8	58.4	12.9
In [13]:

data.head()
Out[13]:
Unnamed: 0	TV	radio	newspaper	sales
0	1	230.1	37.8	69.2	22.1
1	2	44.5	39.3	45.1	10.4
2	3	17.2	45.9	69.3	9.3
3	4	151.5	41.3	58.5	18.5
4	5	180.8	10.8	58.4	12.9
In [14]:

import seaborn as sns
sns.pairplot(data,x_vars=["TV", "radio", "newspaper"], y_vars="sales")
Out[14]:
<seaborn.axisgrid.PairGrid at 0x1129d4908>

In [15]:

sns.pairplot(data,x_vars=["TV", "radio", "newspaper"], y_vars="sales", size = 7, aspect = 0.7, kind='reg')
Out[15]:
<seaborn.axisgrid.PairGrid at 0x112a399e8>

In [16]:

feature_cols = ['TV', 'radio', 'newspaper']
x = data[feature_cols]
x.head()
Out[16]:
TV	radio	newspaper
0	230.1	37.8	69.2
1	44.5	39.3	45.1
2	17.2	45.9	69.3
3	151.5	41.3	58.5
4	180.8	10.8	58.4
In [17]:

y = data['sales']
y.head()
Out[17]:
0    22.1
1    10.4
2     9.3
3    18.5
4    12.9
Name: sales, dtype: float64
In [25]:

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1)
In [26]:

x_train.shape
Out[26]:
(150, 3)
In [27]:

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
In [28]:

linreg.fit(x_train, y_train)
Out[28]:
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
In [29]:

y_pred = linreg.predict(x_test)
In [30]:

from sklearn import metrics
#mean squared error
print (metrics.mean_squared_error(y_test, y_pred))
1.97304562023