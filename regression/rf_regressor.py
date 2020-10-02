import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#read data
data=pandas.read_csv("/home/alex/Desktop/telelis/data/meteo.txt", header=None)

#prepare training test
X_train=data.iloc[:3266,[2,3,4,5,6]]
y_train=data.iloc[:3266,0]

#prepare test set
X_test=data.iloc[-20:,[2,3,4,5,6]]
y_test=data.iloc[-20:,0]

#train rf
regr = RandomForestRegressor(max_depth=500, random_state=42)
regr.fit(X_train, y_train)

#predict
predictions=regr.predict(X_test)

#evaluate
real=y_test
percent_deviations=[]
for i in range(0, 20):
    percent_deviation=abs(real.iloc[i]-predictions[i])/real.iloc[i]*100
    percent_deviations.append(percent_deviation)

mean_percent_deviation=np.mean(percent_deviations)
squarederror=[p**2 for p in percent_deviations]
mse=np.mean(squarederror)
mean_accuracy=100-mean_percent_deviation
print("\nMean Percent Accuracy for Random Forest: %.3f")%mean_accuracy
print("\nMean Percent Deviation for Random Forest: %.3f")%mean_percent_deviation
print("MSE: %.3f")%mse        


# best Acc 95.74, MSE 28.3










