import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


#data
df = pd.read_excel("Retail.xlsx", sheetname=0, header=[0]).iloc[:, :-1]
dc = df.pct_change(periods=1).dropna()

# creates features
X = dc.iloc[:, 2:]
y = dc.iloc[:, 0]

poly = PolynomialFeatures(interaction_only = True, include_bias=False)
xy = poly.fit_transform(X)
y = dc.iloc[:, 0].values

# split
X_train, X_test, y_train, y_test = train_test_split(xy, y,
                                                    random_state=0, test_size=0.1)

# estimation RidgeCV
alphas = np.linspace(0.01, 5, 10)
ridge = RidgeCV(alphas=alphas, normalize=True, cv=5)
lr = ridge.fit(X_train, y_train)
print lr.score(X_test, y_test)

# final esitmate
de = df.tail(2).pct_change(periods=1)
Xe = de.iloc[:, 2:].dropna()

xye = poly.fit_transform(Xe)
pr = lr.predict(xye)

