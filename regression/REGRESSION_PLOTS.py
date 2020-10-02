import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########################################### REGRESSION
##### comparison ar,rf,mlp
n_groups = 3

accuracy= (95.2,95.7,95.9)
mse=(27.5,28.3,24.8)


fig, ax = plt.subplots()
ax.set_yticks([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95])
index = np.arange(n_groups)
bar_width = 0.18

opacity = 0.4

rects1 = plt.bar(index, accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')
rects2 = plt.bar(index +bar_width, mse, bar_width,
                 alpha=opacity,
                 color='r',
                 label='MSE')

                

plt.xlabel('Method')
plt.ylabel('')
plt.title('Best Results for Regression Methods (without MA)')
plt.xticks(index + bar_width / 2, ('Autoregression', 'Random Forest', 'Multi Layer Perceptron'))
plt.legend(loc=7)
plt.tight_layout()
plt.show()
