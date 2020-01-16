import numpy as np

npy_data = np.load('test_video.npy')
print(npy_data.shape)
print(npy_data[0].shape)

# import pickle
#
# f = open('data/ntu/xview/train_label.pkl', 'rb')
# data = pickle.load(f)
#
# for i in range(len(data[0])):
#     print(data[0][i],data[1][i])
#
# print()
# print('len(data)',len(data))
# print('len(data)[0]',len(data[0]))
# print('len(data)[1]',len(data[1]))
# print()
# print(data[0][0], data[1][0])

# print(data)
# wf = open('train_label.txt','w')
# wf.write(str(data))



