import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

'''
X , y = mglearn. datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)

#print(lr.coef_)
#print(lr.intercept_)

print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))
'''

'''

X,y=mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

lr=LinearRegression()
lr.fit(X_train,y_train)

'''

X,y=mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

lr=Ridge()
lr.fit(X_train,y_train)
'''
print(f"{lr.score(X_train,y_train):.2f}")
print(f"{lr.score(X_test,y_test):.2f}")
'''
'''
ridge=Ridge(alpha=0).fit(X_train,y_train)
ridge10=Ridge(alpha=10).fit(X_train,y_train)
ridge01=Ridge(alpha=.1).fit(X_train,y_train)


plt.plot(ridge.coef_,'s',label="Ridge alpha=1")
plt.plot(ridge10.coef_,'^',label="Ridge alpha=10")
plt.plot(ridge01.coef_,'v',label="Ridge alpha=.1")

plt.plot(lr.coef_,'o',label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()

plt.show()
'''

mglearn.plots.plot_ridge_n_samples()
plt.show()