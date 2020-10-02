import pandas
import numpy as np
import matplotlib.pyplot as plt


data=pandas.read_csv("/home/alex/Desktop/telelis/data/meteo.txt")
df=data.iloc[:,0].values

def do_ma(a,b):
    howmany=a #how many steps back to look for average
    num_pred=b #how many predictions to perform
    real=list(df[-num_pred:]) #"test" set
    predictions=[]
    sample=list(df[-howmany-num_pred:-num_pred]) #e.g.: 20 predictions, 10 steps to look back, take elements from last 30th to last 20th
    predictions.append(np.average(sample))
    for i in range(0,num_pred-1):
        sample=sample[1:] #remove 1st element
        sample.append(predictions[i]) #append the previous prediction
        predictions.append(np.average(sample))
    return predictions, real, num_pred



print("RESULTS")
accuracy=[]
mses=[]
for j in range(1,21):
    num=j
    predictions, real, length=do_ma(num,20)
    #find each accuracy and mean accuracy 
    accuracies=[]
    for i in range(0,length):
        temp=abs(100-abs(predictions[i]-real[i])/real[i]*100)
        accuracies.append(temp)
    accuracy.append(np.mean(accuracies))    
    percent_deviations=[100-j for j in accuracies]
    squarederrors=[p**2 for p in percent_deviations]
    mse=np.mean(squarederrors)
    mses.append(mse)
    mean_percent_accuracy=sum(accuracies)/length
    print("MA term: %d,Accuracy: %.3f,MSE: %.3f ")%(num,mean_percent_accuracy, mse)

'''
plt.plot(accuracy, color='blue', label='Accuracy')
plt.legend()
plt.xticks(np.arange(20),['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.xlabel('MA term')
plt.title('MA Accuracy for different terms (ma term=1-20)')
plt.show()

#PLOT MSE
'''
plt.plot(mses, color='red', label='MSE')
plt.legend()
plt.xticks(np.arange(20),['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.xlabel('MA term')
plt.title('MA MSE for different terms (ma term=1-20)')
plt.show()



#best acc 77.7, mse 1091.8

