import pandas
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

data=pandas.read_csv("/home/alex/Desktop/telelis/data/meteo.txt", header=None)
df=list(data.iloc[:,0].values)


############################################# USEFUL FUNCTIONS ##############
#create X_train,y_train, for given Ar_term
def create_datasets(small_df, AR_term):
    X_train=[]
    y_train=[]
    for i in range(AR_term, len(small_df)-AR_term):
        x=small_df[i-AR_term:i+1] 
        y=small_df[i+1]
        X_train.append(x)
        y_train.append(y)
    return X_train, y_train

#train model
def do_training(X_train, y_train):
    mod = linear_model.LinearRegression()
    mod.fit(X_train, y_train)
    return mod

#do_predictions
def do_predictions(model, X_test):
    a=model.predict([X_test])
    return a
    
#update the x_test
def create_xtest(dataset, AR_term):
    a=dataset[-AR_term-1:]
    return a


#########################################START THE ML 
acc=[]
mses=[]
for i in range(1,21):
    #choose configurations
    ar_term=i
    num_predictions=20
    #build the model
    small_df=df[:-num_predictions]
    X_train,y_train=create_datasets(small_df, ar_term)
    model=do_training(X_train, y_train)
    #start the predictions and find errors
    whole_dataset=small_df
    real=small_df[-num_predictions:]
    all_predictions=[]
    percent_deviations=[]
    for i in range(0, num_predictions):
        X_test=create_xtest(whole_dataset, ar_term)
        prediction=do_predictions(model, X_test)
        percent_deviation=abs(real-prediction)/real*100
        percent_deviations.append(percent_deviation[0])
        all_predictions.append(prediction[0])
        whole_dataset.append(prediction[0])
    #find metrics
    accuracies=[100-p for p in percent_deviations]    
    mean_percent_deviation=np.mean(percent_deviations)
    mean_accuracy=np.mean(accuracies)
    squarederror=[p**2 for p in percent_deviations]
    acc.append(mean_accuracy)
    mse=np.mean(squarederror)
    mses.append(mse)
    print("Accuracy: %.3f, MSE: %.3f")%(mean_accuracy, mse)    


plt.plot(acc, color='blue', label='Accuracy')
plt.plot(mses, color='red', label='MSE')
plt.legend(loc=7)
plt.xticks(np.arange(21),['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.xlabel('AR term')
plt.title('Accuracy and MSE for different AR terms (AR term=1-20)')
plt.show()


#best acc 95.2, mse 27.5









