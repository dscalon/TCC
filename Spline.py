#  https://www.analytics-link.com/post/2018/08/17/creating-and-plotting-cubic-splines-in-python

import numpy as np

import matplotlib.pyplot as plt

from scipy import interpolate

timestamp = (0, 5, 10, 15, 30, 35, 40, 50, 55, 60)

distance = (100, 90, 65, 85, 70, 30, 40, 45, 20, 0)

plt.plot(timestamp, distance, 'o')

plt.show()

data = np.array((timestamp, distance))

tck, u = interpolate.splprep(data, s=0)

unew = np.arange(0, 1.01, 0.01)

out = interpolate.splev(unew, tck)

plt.plot(out[0], out[1], color='orange')

plt.plot(data[0, :], data[1, :], 'ob')

plt.show()