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
import matplotlib.animation as animation

fig, ax = plt.subplots() 

th = np.arange(0, 10*np.pi, 0.01)
a=1
b=0.1
r=a*np.exp(b*th)
#r=th**2 #another alternative

x=r*np.cos(th)
y=r*np.sin(th)
#y=y/max(abs(y)) to normalize y values if you want the max value to be 1.0

#plt.plot(x,y) #used to visualize what the spiral would look like before animating.

line, = ax.plot(x,y) 

def animate(i):
    line.set_ydata(r*np.sin(th+i/10.0))
    return line,


# Init only required for blitting to give a clean slate. 
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,


ani = animation.FuncAnimation(fig, animate, np.arange(0, len(x)), init_func=init,
                              interval=25, blit=True) 

plt.show() 

