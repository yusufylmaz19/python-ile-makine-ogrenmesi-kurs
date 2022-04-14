import pandas as ps
import numpy as np
import matplotlib.pyplot as plt

veriler= ps.read_csv('maaslar.csv')

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2]

X=x.values
Y=y.values

from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()