# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 21:40:57 2017

@author: Vikneshan

More Information:
    http://vikneshan.blogspot.com/2017/01/examples-of-animating-plots-in-python.html
    http://stackoverflow.com/questions/21937976/defining-multiple-plots-to-be-animated-with-a-for-loop-in-matplotlib
    
Equivalent to http://matplotlib.org/examples/animation/simple_anim.html, using a for loop

"""

# ** Importing relevant modules
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5*np.pi, 0.01)
plt.axis([min(x),max(x),-1,1])
plt.ion()
y= np.sin(x)
plt.plot(x,y)

for i in range(0,100):
    y=np.sin(x+i/1.0)
    fig,=plt.plot(x, y)
    plt.show()
    plt.pause(0.0001) # changing this doesn't seem to change sine wave travelling speed much, not the most efficient way of plotting
    plt.gcf().clear() # change denominator for i/10 to i or something larger if want faster animation or use matplotlib.animation module
