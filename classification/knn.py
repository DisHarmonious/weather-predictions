import pandas
from sklearn.neighbors import KNeighborsClassifier
import timeit
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

#import training set
dataset=genfromtxt("/home/alex/Desktop/telelis/data/meteo.txt", delimiter=',')
rain=dataset[:,3].tolist()
#find days that there was rain
rainydays=[]
for i in range(0,len(rain)):
    if rain[i]>0:
        rainydays.append(1)
    else:
        rainydays.append(0)
    
#fix training set
X_train=dataset[:3000,[0,1,2,4,5,6]].tolist()
y_train=rainydays[:3000]

#fix test
X_test=dataset[-287:,[0,1,2,4,5,6]].tolist()
y_test=rainydays[-287:]

accuracies=[]
precisions=[]
recalls=[]

print("RESULTS ")
print("k,Accuracy, Precision, Recall")
for i in range(1,21):
    #train
    n=i
    neigh = KNeighborsClassifier(n_neighbors=n, algorithm='auto', weights='distance')
    neigh.fit(X_train, y_train) 
    #predict
    predictions=neigh.predict(X_test)
    #calculate accuracy
    real=y_test
    counter=0
    for i in range(0,len(y_test)):
        if y_test[i]==predictions[i]:
            counter+=1
    accuracy=float(counter)/len(y_test)
    accuracies.append(accuracy)
    #calculate precision and recall
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(0,len(y_test)):
        if real[i]==predictions[i] and real[i]==1:
            tn+=1
        elif real[i]==predictions[i] and real[i]==0:
            tp+=1
        elif real[i]!=predictions[i] and real[i]==1:
            fn+=1
        else:
            fp+=1
    precision=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    recalls.append(recall)
    precisions.append(precision)
    print("%d, %.3f, %.3f, %.3f")%(n, accuracy, precision, recall)

#####PLOT
plt.plot(accuracies, color='blue', label='Accuracy')
plt.plot(precisions, color='red', label='Precision')
plt.plot(recalls, color='green', label='Recall')
plt.legend(loc=3)
plt.xticks(np.arange(20),['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.xlabel('k')
plt.title('kNN Metrics (k=1-20)')
plt.show()

#best acc pr rec : 0.979, 0.976, 0.891