
import pyDOE
import pandas as pd
import numpy as np
import scipy.stats.distributions as dist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from collections import namedtuple
from sklearn.linear_model import ElasticNet
from RBFN import RBFN
from sklearn.preprocessing import MinMaxScaler


ValueRange = namedtuple('ValueRange', ['min', 'max'])

def determinerange(values):
    """Determine the range of values in each dimension"""
    return ValueRange(np.min(values, axis=0), np.max(values, axis=0))


def linearscaletransform(values, *, range_in=None, range_out=ValueRange(0, 1), scale_only=False):
    """Perform a scale transformation of `values`: [range_in] --> [range_out]"""

    if range_in is None:
        range_in = determinerange(values)
    elif not isinstance(range_in, ValueRange):
        range_in = ValueRange(*range_in)

    if not isinstance(range_out, ValueRange):
        range_out = ValueRange(*range_out)

    scale_out = range_out.max - range_out.min
    scale_in = range_in.max - range_in.min

    if scale_only:
        scaled_values = (values / scale_in) * scale_out
    else:
        scaled_values = (values - range_in.min) / scale_in
        scaled_values = (scaled_values * scale_out) + range_out.min

    return scaled_values


''' Elastic Net Regression for Minimax Robustness'''
def elastic_net_minimax(train_data,test_data):
    scaler =  MinMaxScaler().fit(np.r_[train_data.iloc[:,:5].values, test_data.values])
    regr=ElasticNet(alpha= 0.18642169603863867   ,random_state=0 , l1_ratio= 1.0, fit_intercept =True, max_iter=1000,selection='random'
                   ).fit(scaler.transform(train_data.iloc[:,:5]), train_data.iloc[:,6])
    pred = regr.predict(scaler.transform(test_data))
    return regr,pred

''' Elastic Net Regression for Minimax Robustness'''
def elastic_net_composite(train_data,test_data):
    scaler =  MinMaxScaler().fit(np.r_[train_data.iloc[:,:5].values, test_data.values])
    regr=ElasticNet(alpha= 0.23930385822929412      ,random_state=0 , l1_ratio= 1.0, fit_intercept =True, max_iter=1000,selection='random'
                   ).fit(scaler.transform(train_data.iloc[:,:5]), train_data.iloc[:,-1])
    pred = regr.predict(scaler.transform(test_data))
    return regr,pred


''' Kriging for Robust Regularization'''
def kriging_minimax(train_data,test_data):
    kernel =  RBF( 81, (3, 55462) )
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=36,random_state=0,
                                   normalize_y=True ).fit(scaler.transform(train_data.iloc[:,:2]), train_data.iloc[:,3])
    pred = gpr.predict(scaler.transform(test_data))
    return gpr,pred


''' Kriging for Robust Composition'''
def kriging_composite(train_data,test_data):
    kernel =  RBF( 81, (3, 55462) )
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=36,random_state=0,
                                   normalize_y=True ).fit(scaler.transform(train_data.iloc[:,:2]), train_data.iloc[:,-1])
    pred = gpr.predict(scaler.transform(test_data))
    return gpr,pred


''' Random Forest Implementation for Minimax Robustness'''
def rf_minimax(train_data,test_data):
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    regr = RandomForestRegressor(random_state=27,n_estimators=55
                                ).fit(scaler.transform(train_data.iloc[:,:2]), train_data.iloc[:,3])
    pred = regr.predict(scaler.transform(test_data))
    return regr,pred

''' Random Forest Implementation for Composite Robustness'''
def rf_composite(train_data,test_data):
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    regr = RandomForestRegressor(random_state=47,n_estimators=40
                                ).fit(scaler.transform(train_data.iloc[:,:2]), train_data.iloc[:,-1])
    pred = regr.predict(scaler.transform(test_data))
    return regr,pred


''' KNN Regression Implementation for Minimax Robustness'''
def KNN_minimax(train_data,test_data):
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    regr = KNeighborsRegressor(n_neighbors=5,weights='distance',algorithm='brute',p=2
                               ).fit(scaler.transform(train_data.iloc[:,:2]), train_data.iloc[:,3])

    pred = regr.predict(scaler.transform(test_data))
    return regr,pred

''' KNN Regression Implementation for Composite Robustness'''
def KNN_composite(train_data,test_data):
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    regr = KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='brute',p=2
                               ).fit(scaler.transform(train_data.iloc[:,:2]), train_data.iloc[:,-1])

    pred = regr.predict(scaler.transform(test_data))
    return regr,pred

''' Support Vector Regression for Robust Regularization'''
def SVR_minimax(train_data,test_data):
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    gpr = SVR(gamma =  0.981262    ,C =  453.957039, epsilon=0.2,max_iter=1500).fit(
        scaler.transform(train_data.iloc[:,:2]), train_data.iloc[:,3])
    pred = gpr.predict(scaler.transform(test_data))
    return gpr,pred

''' Support Vector Regression for Robust Composition'''
def SVR_composite(train_data,test_data):
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    gpr = SVR(gamma =  4.068846,C = 57.598748, epsilon=0.2,max_iter=1500).fit(
        scaler.transform(train_data.iloc[:,:2]), train_data.iloc[:,-1])
    pred = gpr.predict(scaler.transform(test_data))
    return gpr,pred


''' RBF Network Interpolation for Robust Regularization'''
def RBF_minimax(train_data,test_data):
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    model = RBFN(hidden_shape=51, sigma= 1.421893)
    model.fit(scaler.transform(np.array(train_data.iloc[:,:2])), np.array(train_data.iloc[:,3]))
    pred = model.predict(scaler.transform(np.array(test_data)))
    return model,pred

''' RBF Network Interpolation for Robust Composition'''
def RBF_composite(train_data,test_data):
    scaler = MinMaxScaler().fit(np.r_[train_data.iloc[:,:2].values, test_data.values])
    model = RBFN(hidden_shape=89, sigma= 0.001000)
    model.fit(scaler.transform(np.array(train_data.iloc[:,:2])), np.array(train_data.iloc[:,-1]))
    pred = model.predict(scaler.transform(np.array(test_data)))
    return model,pred
