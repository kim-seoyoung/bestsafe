import numpy as np
import pickle

np_data = np.load('train_data_joint.npy')

# print(np_data)
print(np_data.shape)


with open('train_label.pkl', 'rb') as f:
    pk_lable = pickle.load(f) # 단 한줄씩 읽어옴

print(len(pk_lable))
print(len(pk_lable[0]))
print(len(pk_lable[1]))


#####################################################################
### 3D ###


x_axis = 0
y_axis = 1
z_axis = 2


# point = 16
# new_data_x = np_data[:,point,x_axis]
# new_data_y = np_data[:,point,y_axis]
# new_data_z = np_data[:,point,z_axis]

num = 0
frame = 5
print(pk_lable[0][num],pk_lable[1][num])
new_data_x = np_data[num,x_axis,frame,:,0]
new_data_y = np_data[num,y_axis,frame,:,0]
new_data_z = np_data[num,z_axis,frame,:,0]

#############################################################


# plotting random walk by normal dist.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d') # Axe3D object

sample_size = 50

ax.plot(new_data_x, new_data_y, new_data_z, alpha=0.6, marker='o')
for i in range(25):
    ax.text(new_data_x[i], new_data_y[i], new_data_z[i], i+1)
plt.title("ax.plot")
plt.show()


# plt.close()

# np.reshape(np_data,)




