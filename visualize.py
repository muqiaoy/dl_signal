import numpy as np
import matplotlib.pyplot as plt

data = np.load("/projects/rsalakhugroup/complex/music_test_x_128.npy") 
instance = data[40] 
x = instance[:, 1000, 0]
y = instance[:, 1000, 1] 


times = [i for i in range(len(x))]

plt.scatter(times, x, s=6.25) 
plt.scatter(times, y, s=6.25)
# plt.show()
plt.savefig("music_figure_128.png")