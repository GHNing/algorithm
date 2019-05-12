import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class LR(object):
    def __init__(self,rate = 0.01,iterations=500):
        self.rate = rate
        self.iters = iterations

    def sigmoid(self,a):
        return 1.0/(1+np.exp(-a))

    def run(self,x,y):
        x_matrix = np.mat(x)
        y_matrix = np.mat(y).T
        rows,cols = np.shape(x)
        paras = np.ones((cols,1)).transpose()
        for i in range(self.iters):
            v1 = self.sigmoid(x_matrix * paras.T)
            error = y_matrix - v1
            # print (error)
            paras = paras - self.rate * error.T * x_matrix
            print(paras)

if __name__=='__main__':
    data = datasets.load_iris()
    df_train = pd.DataFrame(data['data'],columns=data['feature_names'])
    Y = pd.Series(data['target'])
    # print(df_train.head())
    # clf =  LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(df_train, Y)
    # print(clf.get_params())
    tmp = LR()
    tmp.run(df_train,Y)