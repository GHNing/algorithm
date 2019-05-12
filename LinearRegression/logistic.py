import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class LR(object):
    def __init__(self,rate = 0.01,iterations=500,):
        self.rate = rate
        self.iters = iterations

    def sigmoid(self,a):
        return 1.0/(1+np.exp(-a))

    def Likelihood_value(self,x,y,paras):
        data = pd.concat([x,y],axis=1)
        count = len(data)
        col = data.columns
        end_value = 1

        for m,n in data.iterrows():
            flag = n[col[-1]]
            p = np.mat(n[col[:-1]]) * paras.T
            hx = self.sigmoid(p.tolist()[0][0])
            if flag == 0.0:
                hx = 1 - hx
                end_value *= hx
            else:
                end_value *= hx

        return end_value

    def run(self,x,y):
        max = 0
        x_matrix = np.mat(x)
        y_matrix = np.mat(y).T
        rows,cols = np.shape(x)
        paras = np.ones((cols,1)).transpose()
        model_panas = np.zeros((cols,1))
        for i in range(10):
            v1 = self.sigmoid(x_matrix * paras.T)
            error = v1 - y_matrix
            # print (error)
            paras = paras - self.rate * error.T * x_matrix
            #迭代是否停止
            tmp = self.Likelihood_value(x,y,paras)
            print(tmp)
            if max > tmp:
                break
            else:
                max = tmp
                model_panas = paras

if __name__=='__main__':
    data = datasets.load_iris()
    df_train = pd.DataFrame(data['data'],columns=data['feature_names'])
    Y = pd.Series(data['target'])
    # print(df_train.head())
    # clf =  LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(df_train, Y)
    # print(clf.get_params())

    t1 = pd.concat([df_train,Y],axis=1)
    t2 = t1.loc[(t1[0] == 0) | (t1[0] == 1),:]
    # print(t2[0].value_counts())
    df_train = t2.iloc[:,:-1]
    Y = t2[0]

    tmp = LR()
    tmp.run(df_train,Y)

    # para = np.ones((1,4))
    # print(tmp.Likelihood_value(df_train,Y,para))