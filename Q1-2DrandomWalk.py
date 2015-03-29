# group assignment 1 Random Walk
# 2D Random Walk
import math
from pylab import *

def walk(x,y):
    # walk function, just one step
    num = random()
    if num < 0.25:
        x = x + 1
        y = y + 1
    elif num < 0.5:
        x = x - 1
        y = y + 1
    elif num < 0.75:
        x = x - 1
        y = y - 1
    else:
        x = x + 1
        y = y - 1
    return x,y

def walker(n):
    # walk n steps
    x = 0
    y = 0
    for i in range(n):
        x, y = walk(x,y)
    return x,y

def avg(n):
    # perform 10000 walkers
    xsum = 0
    x2sum = 0
    r2sum = 0
    x = 0
    y = 0
    for j in range(10000):
        x, y = walker(n)
        xsum += x
        x2sum += x**2
        r2sum = r2sum + x**2 + y**2
    xavg = xsum * 1.0 / 10000
    x2avg = x2sum * 1.0 / 10000
    r2avg = r2sum * 1.0 / 10000
    print str(n) + " steps yield displacement of " + str(xavg)
    return xavg, x2avg, r2avg

nlist = []
xlist = []
x2list = []
r2list = []
for n in range(3, 101):
    nlist.append(n)
    x, x2, r2 = avg(n)
    xlist.append(x)
    x2list.append(x2)
    r2list.append(r2)

plot(nlist, xlist)
xlabel("number of random steps")
ylabel("average displacement from origin")
title("average displacement from origin VS number of random steps")
show()

plot(nlist, x2list)
xlabel("number of random steps")
ylabel("avg of x^2")
title("avg of x^2 VS number of random steps")
show()

plot(nlist, r2list)
xlabel("number of random steps")
ylabel("avg of r^2")
title("avg of r^2 VS number of random steps")
show()
