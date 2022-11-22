import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn import model_selection

boston_dataset=load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target

boston.isnull().sum()

X1=boston[[	'CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']].values
y = boston['MEDV'].values
X1 = (X1 - np.mean(X1,axis=0)) / np.std(X1,axis=0)

X2=boston[['CHAS']].values
X=np.hstack((X1,X2))


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 1234)


from sklearn.svm import SVR
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print (linear_svr.score(X_test, y_test))

Adjusted_R2=1 - (1-linear_svr.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

Adjusted_R2

for i in range(128):

 num_perturb = 500
 X_limePart1 = np.random.normal(0,1,size=(num_perturb,X1.shape[1]))
 dfX_limePart1=pd.DataFrame(X_limePart1,columns=['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
 boston['CHAS'].value_counts()
 perturbations=[]
 for i in range(num_perturb):
  perturbations.append(np.random.choice(2,1,p=[0.93,0.07]))

 LimePart2=pd.DataFrame(perturbations,columns=['CHAS'])

 result = pd.concat([dfX_limePart1, LimePart2], axis=1)
 original_data = np.array(result.iloc[i])
 original_data.reshape(1, 13)
 linear_svr_y_predict2 = linear_svr.predict(result)

 LinearRegression_data = pd.DataFrame(
     columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'CHAS'])
 LinearRegression_data.to_csv('LinearRegression_data.csv', encoding='utf_8_sig')
 tree_data = pd.DataFrame(
     columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'CHAS'])
 tree_data.to_csv('tree_data.csv', encoding='utf_8_sig')
 RandomForestRegressor_data = pd.DataFrame(
     columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'CHAS'])
 RandomForestRegressor_data.to_csv('RandomForestRegressor_data.csv', encoding='utf_8_sig')
 xgb_data = pd.DataFrame(
     columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'CHAS'])
 xgb_data.to_csv('xgb_data.csv', encoding='utf_8_sig')








 import sklearn
 import sklearn.metrics

 distances = sklearn.metrics.pairwise_distances(result.values,original_data.reshape(1,13), metric='cosine').ravel()

 kernel_width = 0.25
 weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function

 from sklearn.linear_model import LinearRegression


 simpler_model = LinearRegression()
 simpler_model.fit(X=result, y=linear_svr_y_predict2, sample_weight=weights)


 simpler_model.coef_2=abs(simpler_model.coef_)

 feature_nameintxt=['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','CHAS_0.0','CHAS_1.0']
 featurecoef=zip(feature_nameintxt,simpler_model.coef_2)
 featurecoef1=dict(featurecoef)

 featurecoef1_new =[featurecoef1]
 LinearRegression_data_new =pd.DataFrame(featurecoef1_new)
 LinearRegression_data_new.to_csv('LinearRegression_data.csv',header=None, mode='a', encoding='utf_8_sig') #保存CSV


    #print(sorted(featurecoef1.items(),key=lambda kv:(kv[1], kv[0])))

    #66行
    # sorted(featurecoef1.items(),key=lambda kv:(kv[1], kv[0]))

 from sklearn import tree

 treemodel=tree.DecisionTreeRegressor(max_depth=5)
 treemodel.fit(result,linear_svr_y_predict2,sample_weight=weights)

 importances = treemodel.feature_importances_

 feature_nameintxt1=['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','CHAS_0.0','CHAS_1.0']
 featureimportances=zip(feature_nameintxt1,importances)
 featureimportances1=dict(featureimportances)

 featureimportances1_new =[featureimportances1]
 tree_data_new =pd.DataFrame(featureimportances1_new)
 tree_data_new.to_csv('tree_data_new.csv',header=None, mode='a', encoding='utf_8_sig') #保存CSV

    #73行
    #sorted(featureimportances1.items(),key=lambda kv:(kv[1], kv[0]))

 from sklearn.ensemble import RandomForestRegressor

 regr_rf=RandomForestRegressor(max_depth=10)
 regr_rf.fit(X=result, y=linear_svr_y_predict2,sample_weight=weights)


 importances2 = regr_rf.feature_importances_


 feature_nameintxt2=['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','CHAS_0.0','CHAS_1.0']
 featureimportances2=zip(feature_nameintxt2,importances2)
 featureimportances3=dict(featureimportances2)

 featureimportances3_new =[featureimportances3]
 RandomForestRegressor_data =pd.DataFrame(featureimportances3_new)
 RandomForestRegressor_data.to_csv('RandomForestRegressor_data.csv',header=None, mode='a', encoding='utf_8_sig') #保存CSV


    #80行
    #sorted(featureimportances3.items(),key=lambda kv:(kv[1], kv[0]))

 import xgboost as xgb

 xgbrModel=xgb.XGBRegressor()
 xgbrModel.fit(result,linear_svr_y_predict2,sample_weight=weights)


 importances3=xgbrModel.feature_importances_
 importances3


 feature_nameintxt3=['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','CHAS_0.0','CHAS_1.0']
 featureimportances3=zip(feature_nameintxt3,importances3)
 featureimportances4=dict(featureimportances3)

 featureimportances4_new =[featureimportances4]
 xgb_data =pd.DataFrame(featureimportances4_new)
 xgb_data.to_csv('xgb_data.csv',header=None, mode='a', encoding='utf_8_sig') #保存CSV

print('所有数据保存完毕！')

