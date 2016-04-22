import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


grid_values = np.arange(0,1.1,0.2)
f, axarr = plt.subplots(6,6, figsize = (60,30),sharex='col', sharey= 'row')

for x,alpha in enumerate(grid_values):
    for y,gamma in enumerate(grid_values):
        data = pd.read_csv("Alpha={}_Gamma={}.csv".format(alpha,gamma))

        #plt.figure(figsize = (28,28))
        axarr[x,y].bar(np.arange(100),data['Net-Reward'], color = 'b')
        axarr[x,y].bar(np.arange(100),data['Reached'], color = 'r')
        axarr[x,y].set_title("Alpha={},Gamma={}".format(alpha,gamma))

f.subplots_adjust(hspace=0.05, wspace = 0.05)
plt.show()
