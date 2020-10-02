import pandas
from sklearn import svm
import timeit
import numpy as np

#import test set
dataset=np.genfromtxt("/home/alex/Desktop/telelis/data/meteo.txt", delimiter=',')
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

#train
clf = svm.SVC()
clf.fit(X_train, y_train) 


#predict
predictions = clf.predict(X_test)

#calculate accuracy
real=y_test
counter=0
for i in range(0,len(y_test)):
    if y_test[i]==predictions[i]:
        counter+=1
accuracy=float(counter)/len(y_test)
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


print("RESULTS")
print("Percent Accuracy of SVM: \n%s ")%accuracy
print("Precision, Recall:")
print(" %.3f, %.3f")%(precision, recall)

#best acc pr rec .94 .976 .745