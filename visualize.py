import numpy as np
import matplotlib.pyplot as plt

data = np.load("/projects/rsalakhugroup/complex/music_test_x_256.npy") 
instance = data[40] 
x = instance[:, 1000, 0]
y = instance[:, 1000, 1] 


times = [i for i in range(len(x))]

plt.scatter(times, x, s=1) 
plt.scatter(times, y, s=1)
# plt.show()
plt.savefig("music_figure.png")