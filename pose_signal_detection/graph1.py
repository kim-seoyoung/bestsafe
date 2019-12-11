import numpy as np


np_data = np.load('data/RGB/npys/a6_s1_t1_color.npy')

# print(np_data)
print(np_data.shape)

#####################################################################
### 3D ###


x_axis = 0
y_axis = 1
z_axis = 2


# point = 16
# new_data_x = np_data[:,point,x_axis]
# new_data_y = np_data[:,point,y_axis]
# new_data_z = np_data[:,point,z_axis]

frame = 30
new_data_x = np_data[frame,:,x_axis]
new_data_y = np_data[frame,:,y_axis]
new_data_z = np_data[frame,:,z_axis]

print(new_data_x)
print(new_data_y)
print(new_data_z)

#############################################################


# plotting random walk by normal dist.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d') # Axe3D object

sample_size = 50

ax.plot(new_data_x, new_data_y, new_data_z, alpha=0.6, marker='o')
for i in range(17):
    ax.text(new_data_x[i], new_data_y[i], new_data_z[i], i + 1)
plt.title("ax.plot")
plt.show()


# plt.close()

# np.reshape(np_data,)




