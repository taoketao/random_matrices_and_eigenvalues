from numpy import e, pi, exp 
import numpy as np
from math import erf
import matplotlib.pyplot as plt
import matplotlib

''' Solutions are y = erf(y)/(sqrt(2*alpha)+(2/sqrt(pi))*e^-y^2) 
    When alpha > 0.138, the only solutions are y=0. '''

buckets = np.linspace(-20,20,93)
def function_of_y(y): 
    return 0.5*( erf(y)/y * 2*y/pi**0.5*exp(-y**2))**2
def difference_function(y,alpha): 
    return y - erf(y)/((2*alpha)**0.5 + 2*pi**-0.5*exp(-y**2))
def difference_function_abs(y,alpha): 
    return abs(y - erf(y)/((2*alpha)**0.5 + 2*pi**-0.5*exp(-y**2)))

#plt_138 = [function_of_y(i) for i in buckets]
#plt.plot(buckets, plt_138)
#plt.show()
#input()

quantity = 41
interval = 17.
if 1:
    alphas = [0.138-x/(quantity*interval) for x in range(1,quantity+1)]\
            +[0.138+x/(quantity*interval) for x in range(1,quantity+1)]
    buckets = np.linspace(-3,3,993)
    plts = [[difference_function(y,alpha) for y in buckets]\
            for alpha in [0.138]+alphas]
if 0:
    alphas = [0.138-0.138*x/quantity for x in range(1,quantity+1)]\
            +[0.138+1.000*x/quantity for x in range(1,quantity+1)]
    buckets = np.linspace(-3.5,3.5,993)
    plts = [[difference_function_abs(y,alpha) for y in buckets]\
            for alpha in [0.138]+alphas]
    plt.yscale('log')
plt.plot(buckets, [0]*len(buckets), color='black')
plt.plot(buckets, plts[0], color=(0,0,1))
for i in range(quantity):
#    plt.plot(buckets, plts[i+1], color=(1.*i/quantity,1.-i/quantity,0.7))
    plt.plot(buckets, plts[i+1], color=(0,1.-i/quantity,0.7))
for i in range(quantity):
#    plt.plot(buckets, plts[i+quantity+1], color='green')
#    plt.plot(buckets, plts[i+quantity+1], color=(1.-i/quantity,1.*i/quantity,0.7))
    plt.plot(buckets, plts[i+quantity+1], color=(1.-i/quantity,0,0.7))
plt.show()
#input()
plt.close()

import sys;sys.exit()

color = plt.cm.hsv(0.3) # r is 0 to 1 inclusive
line = matplotlib.lines.Line2D(buckets, plts[0], color=color)
plt.plot(buckets, line)
plt.show()
plt.close()
