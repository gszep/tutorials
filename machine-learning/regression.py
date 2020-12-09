from numpy import *
from pandas import *
from matplotlib.pyplot import *

from sklearn.linear_model import LinearRegression

df = read_csv('data/global-temperatures.tsv', sep='\t')
X = df['Age My'].values
y = df['Ts'].values

regressor = LinearRegression()
regressor.fit(X.reshape(-1,1),y.reshape(-1,1))

plot(X,y)

from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as linear_model


from sklearn.pipeline import Pipeline
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
    ('linear', linear_model.LinearRegression(fit_intercept=False))])

model.fit(X[::100].reshape(1,-1),y[::100].reshape(1,-1))