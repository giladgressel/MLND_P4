import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


grid_values = np.arange(0,1.1,0.2)
f, axarr = plt.subplots(6,6, figsize = (60,60),sharex='col', sharey= 'row')

for x,alpha in enumerate(grid_values):
    for y,gamma in enumerate(grid_values):
        data = pd.read_csv("Alpha={}_Gamma={}.csv".format(alpha,gamma))
        axarr[x,y].bar(np.arange(100),data['Net-Reward'], color = 'b')
        axarr[x,y].bar(np.arange(100),data['Reached'], color = 'r')
        axarr[x,y].text(.5,.9,"Alpha={},Gamma={}".format(alpha,gamma),horizontalalignment='center',
                        transform=axarr[x,y].transAxes, size = 40)
        axarr[x,y].tick_params(axis = 'both', which='both', labelsize = 25)


f.subplots_adjust(hspace=0.05, wspace = 0.05)
plt.show()
