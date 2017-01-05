"""
A simple example of an animated plot

Source:http://matplotlib.org/examples/animation/simple_anim.html
  
More Information:http://vikneshan.blogspot.com/2017/01/examples-of-animating-plots-in-python.html

> - added commentary
"""

# > Importing relevant modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots() # > returns tuple containing figure and axes, something like a handle or an object

x = np.arange(0, 2*np.pi, 0.01) # > initializes x values for t=0
line, = ax.plot(x, np.sin(x)) # > returns a tuple with one element - contains x and y values, and more attributes

# > feeds the x values at t=i/10 or 0.1 increments for this example t=0 to t=19.9
def animate(i):
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,


# Init only required for blitting to give a clean slate. 
def init(): #> Basically reseting the figure/plot so new wave can be plotted to create the illusion of motion
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

# > Makes an animation by repeatedly calling a function func, passing in (optional) arguments.
ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=25, blit=True) #> function to animate plot, refer http://matplotlib.org/api/animation_api.html

plt.show() # > displays plot
