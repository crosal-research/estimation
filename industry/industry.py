import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


#data
df = pd.read_excel("Industry.xlsx", sheetname=0, header=[0],
                   parse_cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], skiprows=[0, 1], 
                   index_col=[0]).iloc[:, :-1]
dc = df.iloc[:, [0, 1, 2, 4, 5, 6, 7]].pct_change(periods=1)
dc['DU'] = df['DU']
dc['ULC'] = df['ULC']
dc.dropna(inplace=True, subset=['ABPO'])

# creates features
dobs = dc.iloc[:-1,].dropna()
X = dobs.iloc[:, [1, 2, 3, 5, 6, 7]]
y = dobs.iloc[:, 0]

#poly = PolynomialFeatures(interaction_only = True, include_bias=False)
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


# estimation LassoCV
# alphas = np.linspace(0.01, 5, 30)
# lasso = LassoCV(alphas=alphas, normalize=True, cv=5)
# ll = lasso.fit(X_train, y_train)
# print ll.score(X_test, y_test)

# estimation Kneighbors
# neighbors = KNeighborsRegressor(n_neighbors=3)
# ln = neighbors.fit(X_train, y_train)
# print ln.score(X_test, y_test)

# estimation RandomForest
tree = RandomForestRegressor(max_depth=3, random_state=0)
lt = tree.fit(X_train, y_train)
print lt.score(X_test, y_test)


# final esitmate
de = dc.tail(2).pct_change(periods=1)
Xe = de.iloc[:, [1, 2, 3, 5, 6, 7]].dropna()

xye = poly.fit_transform(Xe)
pr = lt.predict(xye)

