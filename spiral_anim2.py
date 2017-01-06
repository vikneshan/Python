# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:34:21 2017

@author: Vikneshan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

a=1
b=0.1
th=np.arange(0,10*np.pi,0.1)
r=a*np.exp(b*th)
x=r*np.cos(th)
y=r*np.sin(th)
x=x[::-1]
y=y[::-1]

fig, ax = plt.subplots()
line, = ax.plot(x,y)


def animate(i):
    line.set_xdata(x[0:i])
    line.set_ydata(y[0:i])  # update the data
    return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x)), init_func=init,
                              interval=25, blit=True)
plt.show()