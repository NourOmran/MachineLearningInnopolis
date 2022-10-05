#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler , StandardScaler ,RobustScaler,PolynomialFeatures , LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight


# In[291]:


bitrate_train_df = pd.read_csv('bitrate_train.csv')
bitrate_test_df  = pd.read_csv('bitrate_test.csv')
print(bitrate_train_df.shape)
print(bitrate_test_df.shape)
bitrate_train_df.head()




poly = PolynomialFeatures(interaction_only=True)

arr=np.array(poly.fit_transform(bitrate_train_df))
train_poly=pd.DataFrame(arr)

arr1 = np.array(poly.fit_transform(bitrate_test_df))
test_poly=pd.DataFrame(arr1)


# In[294]:


bitrate_train_df.describe().T


# In[295]:




# In[296]:




# # Data preprocessing and visualization

# 1-check for null values and categorical first for the train set then test set 

# In[297]:



def check_null_categorical(data):
    types=data.dtypes
    print(f"Number categorical featues: {sum(types=='object')} \n")
    if sum(types=='object') ==0:
        print(f"Number of null values : {np.sum(np.sum(np.isnan(data)))}\n")
check_null_categorical(bitrate_train_df)
check_null_categorical(bitrate_test_df)


# In[6]:


bitrate_train_df.boxplot(grid=False,fontsize=15 , figsize=(80,80))


# In[298]:


def Data_pre(train,test,mode_ofscaling):
    print(mode_ofscaling)
    #column = ['fps_mean','fps_std','rtt_mean','rtt_std','dropped_frames_mean','dropped_frames_std','dropped_frames_max','bitrate_mean','bitrate_std','target']
    
    type_of_scaler=[StandardScaler(),MinMaxScaler(),RobustScaler()]
    
    if str(mode_ofscaling) in str(type_of_scaler):
        
        scaler = mode_ofscaling
        scaler.fit(train)
        scaled_data = scaler.transform(train)
        X_train_scaled= pd.DataFrame(scaled_data)
        X_train_scaled.head()
        
        scaler1 = mode_ofscaling
        scaler1.fit(test)
        scaled_data1 = scaler1.transform(test)
        X_test_scaled= pd.DataFrame(scaled_data1)
        X_test_scaled.head()
        X_train =X_train_scaled.iloc[:,:-1]
        y_train =X_train_scaled.iloc[:, -1]
        X_test = X_test_scaled.iloc[:,:-1]
        y_test = X_test_scaled.iloc[:, -1]
        
        return X_train,y_train,X_test,y_test
    
    
    if mode_ofscaling=='NoScale':
        
        X_train =train.iloc[:,:-1]
        y_train =train.iloc[:, -1]
        X_test = test.iloc[:,:-1]
        y_test = test.iloc[:, -1]
        
        return X_train,y_train,X_test,y_test


# In[304]:


def plot_Model_info(model,y_test,y_pred):
    print(model)
    if str(model)==str(LinearRegression()):
        print(f"Model intercept : {model.intercept_}")
        print(f"Model coefficient : {model.coef_}")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('r2_score :',metrics.r2_score(y_test, y_pred))
    print('-'*100)


# In[300]:


def plot_Model_info_classification(model,y_test,y_pred,x_test):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    print('Testing accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
    print('Testing precision = {}'.format(metrics.precision_score(y_test, y_pred)))
    print('Testing recall = {}'.format(metrics.recall_score(y_test ,y_pred)))
    print('F1 score ={}' .format(metrics.f1_score(y_test ,y_pred)))
    print('weighted F1 score ={}' .format(metrics.f1_score(y_test ,y_pred,average='weighted')))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
                               
    disp.plot()
    plt.show()


# In[301]:


def best_alpha(model,x_train_lasso,y_train_lasso,x_val_lasso,y_val_lasso):
    alphas =[10,5,3,2.9,2.4,2.3,2.2, 2, 1.5, 1.3, 1.2, 1.1, 1, 0.3, 0.2,0.15,0.11,0.1,0.01,0.001,0.00001,0.000001,0.0000001]
    losses = []
    if str(model)=='Ridge()': 
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(x_train_lasso, y_train_lasso)
            y_pred = ridge.predict(x_val_lasso)
            mse = metrics.mean_squared_error(y_val_lasso, y_pred)
            losses.append(mse)
        best_alpha = alphas[np.argmin(losses)]
        print("Best value of alpha:", best_alpha)
        return best_alpha
    
    if str(model)=='Lasso()':
            
        for alpha in alphas:
            lasso = Ridge(alpha=alpha)
            lasso.fit(x_train_lasso, y_train_lasso)
            y_pred = lasso.predict(x_val_lasso)
            mse = metrics.mean_squared_error(y_val_lasso, y_pred)
            losses.append(mse)
        best_alpha = alphas[np.argmin(losses)]
        print("Best value of alpha:", best_alpha)
        return best_alpha


# In[302]:


def do_pca(x_train,x_test):
        pca = PCA(n_components=50)
        
        x_train_reduced = pca.fit_transform(x_train)
        
        cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
        
        plt.figure(figsize=(9, 6))
        
        plt.bar(range(50), pca.explained_variance_ratio_, alpha=0.5, align='center',label='individual explained variance')
                
        plt.step(range(50), cum_var_exp, where='mid',label='cumulative explained variance')          
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        x_test_reduced = pca.transform(x_test)
        return x_train_reduced,x_test_reduced
        


# In[306]:


def Model(model,pca,n_com,allow_poly,classification):
    
    if allow_poly and  (not classification):
        x_train,y_train,x_test,y_test=Data_pre(train_poly,test_poly,StandardScaler())
        
        
    elif (not allow_poly) and (not classification): 
        x_train,y_train,x_test,y_test=Data_pre(bitrate_train_df,bitrate_test_df,StandardScaler())
    
    elif (not allow_poly) and classification:
        scaler=StandardScaler()
        scaler.fit(stream_quality_train.iloc[:,:-1])
        scaled_data = scaler.transform(stream_quality_train.iloc[:,:-1])
        X_train_scaled= pd.DataFrame(scaled_data)
        x_train=X_train_scaled
        y_train=stream_quality_train.iloc[:,-1]
        print(y_train.shape)
        print(x_train.shape)
        
        scaler1=StandardScaler()
        scaler1.fit(stream_quality_test.iloc[:,:-1])
        scaled_data1 = scaler1.transform(stream_quality_test.iloc[:,:-1])
        X_test_scaled= pd.DataFrame(scaled_data1)
        x_test=X_test_scaled
        y_test=stream_quality_test.iloc[:,-1]
        
       
    if pca:
        x_train,x_test=do_pca(x_train,x_test)
        x_train=pd.DataFrame(x_train)
        x_test=pd.DataFrame(x_test)
        
 
    
    
    if str(model)==str(LinearRegression()):
        if pca: 
            dem=n_com
        else:
            dem=56 
        model.fit(x_train.iloc[:,:dem], y_train)
        y_pred= model.predict(x_test.iloc[:,:dem])
        print(cross_val_score(model, x_train.iloc[:,:dem], y_train, cv=5))
        plot_Model_info(model,y_test,y_pred)
        

    
    if str(model)==str(Lasso()):
        if pca: 
            dem=n_com
        else:
            dem=56
        lasso = Lasso()
        lasso.fit(x_train.iloc[:,:dem], y_train)
        lasso = Lasso(best_alpha(Lasso(),x_train.iloc[:,:dem],y_train,x_test.iloc[:,:dem],y_test))
        lasso.fit(x_train.iloc[:,:dem], y_train)
        y_pred = lasso.predict(x_test.iloc[:,:dem])
        print(cross_val_score(lasso, x_train.iloc[:,:dem], y_train, cv=5))
        plot_Model_info(model,y_test,y_pred)
       
   
    if str(model)==str(Ridge()):
        if pca: 
            dem=n_com
        else:
            dem=56
        ridge = Ridge()
        ridge.fit(x_train.iloc[:,:dem], y_train)
        ridge = Ridge(best_alpha(Ridge(),x_train.iloc[:,:dem],y_train,x_test.iloc[:,:dem],y_test))
        ridge.fit(x_train.iloc[:,:dem], y_train)
        y_pred = ridge.predict(x_test.iloc[:,:dem])
        plot_Model_info(model,y_test,y_pred)
    
    if str(model)==str(LogisticRegression()):
        if pca: 
            dem=n_com
        else:
            dem=11
        
        
        Logistic = LogisticRegression(random_state=42).fit(x_train.iloc[:,:dem], y_train)
        y_pred = Logistic.predict(x_test.iloc[:,:dem])
        plot_Model_info_classification(Logistic,y_test,y_pred,x_test.iloc[:,:dem])
        
        print('-'*100)
        print('with L1\n')
        Logistic1 = LogisticRegression(penalty='l1', solver='liblinear',random_state=42).fit(x_train.iloc[:,:dem], y_train)
        y_pred = Logistic1.predict(x_test.iloc[:,:dem])
        plot_Model_info_classification(Logistic1,y_test,y_pred,x_test.iloc[:,:dem])
        
    if str(model)==str(LogisticRegression(class_weight={0:0.53674572,1:7.30351369})):
        if pca: 
            dem=n_com
        else:
            dem=11
        Logistic = LogisticRegression(class_weight={0:0.53674572,1:7.30351369},random_state=42).fit(x_train.iloc[:,:dem], y_train)
        y_pred = Logistic.predict(x_test.iloc[:,:dem])
        plot_Model_info_classification(Logistic,y_test,y_pred,x_test.iloc[:,:dem])
        
        print('-'*100)
        print('with L1\n')
        Logistic1 = LogisticRegression(class_weight={0:0.53674572,1:7.30351369},penalty='l1', solver='liblinear',random_state=42).fit(x_train.iloc[:,:dem], y_train)
        y_pred = Logistic1.predict(x_test.iloc[:,:dem])
        plot_Model_info_classification(Logistic1,y_test,y_pred,x_test.iloc[:,:dem])

        


# # with no polynomail features and no pca 

# In[308]:


Model(Ridge(),pca=False,n_com=37,allow_poly=False,classification=False )
Model(Lasso(),pca=False,n_com=37,allow_poly=False,classification=False )
Model(LinearRegression(),pca=False,n_com=37,allow_poly=False,classification=False )


# # with polynomail 
# 
# 

# In[309]:


Model(Ridge(),pca=False,n_com=37,allow_poly=True,classification=False )
Model(Lasso(),pca=False,n_com=37,allow_poly=True,classification=False )
Model(LinearRegression(),pca=False,n_com=37,allow_poly=True,classification=False )


# # Ridge with PCA

# In[310]:


Model(LinearRegression(),pca=True,n_com=37,allow_poly=True,classification=False )


# In[199]:


stream_quality_train = pd.read_csv('train_data.csv')
stream_quality_test  = pd.read_csv('test_data.csv')


# In[200]:


print(stream_quality_train.shape)
print(stream_quality_test.shape)
stream_quality_train.head()


# In[201]:


print(check_null_categorical(stream_quality_train))
print(np.unique(stream_quality_train['auto_fec_state'],return_counts=True ))
print(np.unique(stream_quality_train['auto_bitrate_state'],return_counts=True))
print(np.unique(stream_quality_train['stream_quality'],return_counts=True))


# #2 tha dataset is ubalanced 378738 for class 0 and 27834 for class 1 

# In[202]:


label =LabelEncoder()
stream_quality_train['auto_bitrate_state']=np.array(label.fit_transform(stream_quality_train['auto_bitrate_state']))
stream_quality_train['auto_fec_state']=np.array(label.fit_transform(stream_quality_train['auto_fec_state']))


stream_quality_test['auto_bitrate_state']=np.array(label.fit_transform(stream_quality_test['auto_bitrate_state']))
stream_quality_test['auto_fec_state']=np.array(label.fit_transform(stream_quality_test['auto_fec_state']))


# In[203]:



stream_quality_train=pd.DataFrame(stream_quality_train)


# In[184]:







# In[311]:


Model(LogisticRegression(),pca=False,n_com=0,allow_poly=False,classification=True )


# # A low recall score (<0.5) means your classifier has a high number of False negatives which can be an outcome of imbalanced class or untuned model hyperparameters. In an imbalanced class problem

# In[312]:


from sklearn.utils.class_weight import compute_class_weight

w=compute_class_weight(class_weight='balanced',classes=np.unique(stream_quality_train.iloc[:,-1]),y=stream_quality_train.iloc[:,-1])
print(w)


# In[313]:


Model(LogisticRegression(class_weight={0:0.53674572,1:7.30351369}),pca=False,n_com=0,allow_poly=False,classification=True )


# # The Best result till now is L1 with class weight 

# # know we will check the outliers 

# In[266]:


stream_quality_train.boxplot(column=['auto_fec_mean'],fontsize=15 , figsize=(20,20))
#boxplot = df.boxplot(column=['Col1', 'Col2'], return_type='axes')
#>>> type(boxplot)


# # for ourliers we will use Isolation Forest Algorithm.

# In[269]:


from sklearn.ensemble import IsolationForest
clf = IsolationForest(random_state=0).fit_predict(stream_quality_train.iloc[:,:-1],stream_quality_train.iloc[:,-1])


# In[271]:


print(clf.shape)


# # Returns -1 for outliers and 1 for inliers.

# In[272]:


outlier_index=[]
for i in range(len(clf)):
    if clf[i]==-1:
        outlier_index.append(i)
print(len(outlier_index)) 


# In[273]:


stream_quality_train.shape


# In[274]:


stream_quality_train=stream_quality_train.drop(outlier_index)
stream_quality_train.shape


# In[286]:


Model(LogisticRegression(),pca=False,n_com=0,allow_poly=False,classification=True )


# In[287]:


from sklearn.utils.class_weight import compute_class_weight

w=compute_class_weight(class_weight='balanced',classes=np.unique(stream_quality_train.iloc[:,-1]),y=stream_quality_train.iloc[:,-1])
print(w)


# the new weight is [ 0.52012869 12.92008487]

# In[288]:


scaler=StandardScaler()
scaler.fit(stream_quality_train.iloc[:,:-1])
scaled_data = scaler.transform(stream_quality_train.iloc[:,:-1])
X_train_scaled= pd.DataFrame(scaled_data)
x_train=X_train_scaled
y_train=stream_quality_train.iloc[:,-1]
print(y_train.shape)
print(x_train.shape)

scaler1=StandardScaler()
scaler1.fit(stream_quality_test.iloc[:,:-1])
scaled_data1 = scaler1.transform(stream_quality_test.iloc[:,:-1])
X_test_scaled= pd.DataFrame(scaled_data1)
x_test=X_test_scaled
y_test=stream_quality_test.iloc[:,-1]
Logistic = LogisticRegression(class_weight={0:0.52012869,1:12.92008487},random_state=42).fit(x_train.iloc[:,:-1], y_train)
y_pred = Logistic.predict(x_test.iloc[:,:-1])
plot_Model_info_classification(Logistic,y_test,y_pred,x_test.iloc[:,-1])

print('-'*100)
print('with L1\n')
Logistic1 = LogisticRegression(class_weight={0:0.52012869,1:12.92008487},penalty='l1', solver='liblinear',random_state=42).fit(x_train.iloc[:,:-1], y_train)
y_pred = Logistic1.predict(x_test.iloc[:,:-1])
plot_Model_info_classification(Logistic1,y_test,y_pred,x_test.iloc[:,:-1])


# In[290]:


scaler=StandardScaler()
scaler.fit(stream_quality_train.iloc[:,:-1])
scaled_data = scaler.transform(stream_quality_train.iloc[:,:-1])
X_train_scaled= pd.DataFrame(scaled_data)
x_train=X_train_scaled
y_train=stream_quality_train.iloc[:,-1]
print(y_train.shape)
print(x_train.shape)

scaler1=StandardScaler()
scaler1.fit(stream_quality_test.iloc[:,:-1])
scaled_data1 = scaler1.transform(stream_quality_test.iloc[:,:-1])
X_test_scaled= pd.DataFrame(scaled_data1)
x_test=X_test_scaled
y_test=stream_quality_test.iloc[:,-1]
Logistic = LogisticRegression(class_weight={0:0.52012869,1:12.92008487},random_state=42).fit(x_train.iloc[:,:-1], y_train)
y_pred = Logistic.predict(x_test.iloc[:,:-1])
plot_Model_info_classification(Logistic,y_test,y_pred,x_test.iloc[:,-1])

print('-'*100)
print('with L2\n')
Logistic1 = LogisticRegression(class_weight={0:0.52012869,1:12.92008487},penalty='l2', solver='liblinear',random_state=42).fit(x_train.iloc[:,:-1], y_train)
y_pred = Logistic1.predict(x_test.iloc[:,:-1])
plot_Model_info_classification(Logistic1,y_test,y_pred,x_test.iloc[:,:-1])


# # trying with svm

# In[282]:


from sklearn import svm
clf = svm.SVC()
clf.fit(x_train.iloc[:,:-1], y_train)
y_pred = clf.predict(x_test.iloc[:,:-1])
plot_Model_info_classification(Logistic1,y_test,y_pred,x_test.iloc[:,:-1])


# In[285]:


plot_Model_info_classification(Logistic1,y_test,y_pred,x_test.iloc[:,:-1])


# In[ ]:




