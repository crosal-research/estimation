import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


#data
df = pd.read_excel("ibc_br.xlsx", sheetname=0, header=[0],
                   parse_cols=[0, 1, 2, 3, 4, 5, 6], skiprows=[0, 1], 
                   index_col=[0])
df.columns = ['ibc_br', 'industry', 'commerce', 'service', 'commerceT', "labor"]
dc = df.pct_change(periods=1)*100

# creates features
dobs = dc.iloc[:-1,:].dropna()
X = dobs.iloc[:, 1:]
y = dobs.iloc[:, 0]

poly = PolynomialFeatures(interaction_only = False, include_bias=False)
xy = poly.fit_transform(X)


# split
X_train, X_test, y_train, y_test = train_test_split(xy, y,
                                                    random_state=0, test_size=0.1)

# estimation RidgeCV
alphas = np.linspace(0.01, 5, 30)
ridge = RidgeCV(alphas=alphas, normalize=True, cv=5)
lr = ridge.fit(X_train, y_train)
print lr.score(X_test, y_test)

# estimation lassoCV
alphas = np.linspace(0.01, 5, 30)
lasso = LassoCV(alphas=alphas, normalize=True, cv=5)
ll = lasso.fit(X_train, y_train)
print ll.score(X_test, y_test)


# final esitmate
de = dc.tail(3).pct_change(periods=1)
Xe = de.iloc[[1], 1:].dropna()

xye = poly.fit_transform(Xe)
pr = ll.predict(xye)

