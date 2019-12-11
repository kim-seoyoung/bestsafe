import numpy as np

# detections_path = 'data_2d_' + 'custom' + '_' + 'myvideos' + '.npz'
#
# # Load serialized dataset
# data = np.load(detections_path, allow_pickle=True)
# resolutions = data['metadata'].item()['video_metadata']
#
# print(data['metadata'])


# np_data = np.load('interpolation_test/RGB/ArmCrossNpy/a6_s1_t1_color.npy')
# print(np_data.shape)
#
# np_data2 = np.load('output/a6_s1_t1_color2.npy')
# print(np_data2.shape)
#
# test = ''
# if test:
#     print('True')
# else:
#     print('False')
#
# np_data3 = np.load('a6_s1_t1_color.npy')
# print(np_data3[0,:,:])


## npy, pkl 파일 검증 ##

np_data = np.load('data/mydata/gen/xsub/train_data_joint.npy')
np_data_val = np.load('data/mydata/gen/xsub/val_data_joint.npy')
print(np_data.shape)
print(np_data[0][0][0])
print(np_data_val.shape)


import pickle

with open('data/mydata/gen/xsub/train_label.pkl', 'rb') as f:
    train_label = pickle.load(f)

with open('data/mydata/gen/xsub/val_label.pkl', 'rb') as f:
    val_label = pickle.load(f)


print('train_label', train_label[0][0],train_label[1][0])
print('val_label', val_label[0][0],val_label[1][0])


