# MachineLearningInnopolis
A machine learning  repository for machine learning tasks ( Masters )

## Data_pre
### Takes The DataFrame and the mood of scaling the mode of scaling is StandardScaler(),MinMaxScaler(),RobustScaler()] or just pass no scaling 
```python
def Data_pre(train,test,mode_ofscaling): 


# return X_train,y_train,X_test,y_test
```




## plot_Model_info(model,y_test,y_pred):
```python
 #polt the regression model
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
```

## pplot_Model_info_classification(model,y_test,y_pred):
```python
### polt the classification model

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


```
## best_alpha
```python
#return best alhps 
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
   ``` 


## do_pca
 
 ```python
      # retuen redcude after making tranform 
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
        
   ``` 
   
## Model 
 ```python
 #type of model , aloow pca or not , allow polynomial or , is it classification task or not 
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
        
        
        
  #Model(Ridge(),pca=False,n_com=37,allow_poly=False,classification=False )
   #Model(Lasso(),pca=False,n_com=37,allow_poly=False,classification=False )
    #Model(LinearRegression(),pca=False,n_com=37,allow_poly=False,classification=False )
 ``` 
        
