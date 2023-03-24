# Machine Learning

# Regression

# Classification

# Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

hare_speed = 2
maxval =10
Interval = maxval +1

h_xdata, h_ydata = [], []
plt.figure(figsize=(10,10))

for t in np.linspace(0, maxval, Interval):
    h_y = hare_speed*t
    h_xdata.append(t)
    h_ydata.append(h_y)

plt.plot(h_xdata, h_ydata, 'r', label='Hare')
plt.title('linear regression', fontsize= 6)
plt.xlabel('time', fontsize= 6)
plt.ylabel('distance', fontsize= 6)
plt.legend()

plt.show()

# Hypothesis

velocity_variance = 0.2
LINES = 5



a_val = hare_speed + (velocity_variance * LINES)
h_xdata, h_ydata, v_xdata, v_ydata = [], [], [], []


for t in np.linspace(0, maxval, Interval):
    h_y = hare_speed*t
    h_xdata.append(t)
    h_ydata.append(h_y)
    a = a_val - (t * velocity_variance )
    for i in np.linspace(0, maxval, Interval):
        h_y = a * i
        v_xdata.append(t)
        v_ydata.append(h_y)
    plt.plot(v_xdata, v_ydata, 'b', alpha = 0.2)

plt.plot(h_xdata, h_ydata, 'r', label='Hare')
plt.title('linear regression', fontsize= 6)
plt.xlabel('time', fontsize= 6)
plt.ylabel('distance', fontsize= 6)
plt.legend()

plt.show()
