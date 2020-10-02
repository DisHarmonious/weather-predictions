import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##### DT
n_groups = 3

accuracy= (0.99,0.98,0.94)
precision=(0.97,0.98,0.976)
recall = (0.97,0.90, 0.74)


fig, ax = plt.subplots()
ax.set_yticks([.5,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95])
index = np.arange(n_groups)
bar_width = 0.18

opacity = 0.4

rects1 = plt.bar(index, accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')
rects2 = plt.bar(index +bar_width, precision, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Precision')
rects3 = plt.bar(index+bar_width*2, recall, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Recall')
              

plt.xlabel('Problem')
plt.ylabel('')
plt.title('Comparison of Best Performance for Classification Methods')
plt.xticks(index + bar_width / 2, ('Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine'))
plt.legend(loc=3)
plt.tight_layout()
plt.show()
