from rltools.theanotools import Theano_RBF_Projector
from rltools.MountainCar import MountainCar

import matplotlib.pyplot as plt
import numpy as np


domain = MountainCar(random_start=True, max_episode=1000)
s_range = domain.state_range

xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                         np.linspace(s_range[0][1], s_range[1][1], 10))
       
centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
widths = (s_range[1]-s_range[0])*0.085

phi = Theano_RBF_Projector(centers, widths, False, True)

xx, yy = np.meshgrid(np.linspace(-4, 4, 100, True),
                     np.linspace(-4, 4, 100, True))

points = np.hstack((xx.reshape((-1,1)),
                    yy.reshape((-1,1))))

grid = phi(points)

plt.figure()
plt.pcolormesh(xx, yy, grid.dot(np.hstack((np.ones(50), -np.ones(50)))).reshape((100,-1)))
# 
# plt.figure()
# plt.pcolormesh(xx, yy, grid.dot([1,0,0]).reshape((100,-1)))
# 
# plt.figure()
# plt.pcolormesh(xx, yy, grid.dot([0,1,0]).reshape((100,-1)))
# 
# plt.figure()
# plt.pcolormesh(xx, yy, grid.dot([0,0,1]).reshape((100,-1)))

plt.show()