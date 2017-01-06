# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:16:29 2017

@author: Vikneshan

A simple example to create a spiral animation

Reference:
    1)http://mathworld.wolfram.com/LogarithmicSpiral.html
    2)http://www.matlab-cookbook.com/recipes/0050_Plotting/0030_Line_and_Scatter_Plots/plottingASpiral.html
"""

import numpy as np
import matplotlib.pyplot as plt
a=1
b=0.1
th=np.arange(0,10*np.pi,0.1)
r=a*np.exp(b*th)
x=r*np.cos(th)
y=r*np.sin(th)
x=x[::-1]
y=y[::-1]

x_lb=min(x)
x_ub=max(x)
y_lb=min(y)
y_ub=max(y)



#color=['ro','go','bo','ko','co','mo','yo'] #circle marker colors
color=['r','g','b','k','c','m','y'] #line color
col_count=0
plt.ion()

for i in range(0,len(th)):
    if col_count>6:
        col_count=0
        
    fig,=plt.plot(x[0:i], y[0:i],color[col_count])
    fig.axes.set_xlim([x_lb,x_ub])
    fig.axes.set_ylim([y_lb,y_ub])   
    plt.show(fig)
    plt.pause(0.05) # changing this doesn't seem to change sine wave travelling speed much, not the most efficient way of plotting
    plt.gcf().clear() 
    col_count+=1


