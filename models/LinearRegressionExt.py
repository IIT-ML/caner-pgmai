'''
Created on Feb 21, 2015

@author: ckomurlu
'''
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class LinearRegressionExt(LinearRegression):
    def __init__(self, *args, **kwargs):
        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            del kwargs['degree']
            self.poly = PolynomialFeatures(degree=self.degree)
        else:
            self.degree = 1
        super(self.__class__,self).__init__(*args, **kwargs)
        
#     def fit(self):
#         raise NotImplemented()

    def fit(self, X, y, n_jobs=1):
        if self.degree > 1:
            X = self.poly.fit_transform(X)
        super(self.__class__,self).fit(X, y, n_jobs=n_jobs)
        
    def predict(self, X):
        if self.degree > 1:
            X = self.poly.fit_transform(X)
        return LinearRegression.predict(self, X)
    