
import numpy as np

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
from common.visualization import render_animation
from common.custom_dataset import CustomDataset


file_path = 'input/a6_s1_t1_color.mp4.npy'

file_name = file_path.split('/')[-1].split('.')[0]
np_load_file_path = 'input/' + file_name + '.npy'
viz_subject = file_name + '.mp4'
# viz_video = 'data/videos/' + file_name + '.mp4'
viz_video = None
viz_export = 'output/' + file_name + '2.npy'
viz_output = 'output/' + file_name + '.gif'


viz_action = 'custom'
viz_camera = 0

dataset_name = 'custom'
keypoints_name = 'myvideos'
myvideos_path = 'input/data_2d_' + dataset_name + '_' + keypoints_name + '.npz'

subjects_train = 'S1,S5,S6,S7,S8'.split(',')
subjects_test = 'S9,S11'.split(',')
subjects_semi = []
render = True
downsample = 1 #'downsample frame rate by factor (semi-supervised)'
actions = '*'
architecture = '3,3,3,3,3' # default 3,3,3
disable_optimizations = True
dense = False
test_time_augmentation = True

stride = 1
epochs = 60
batch_size = 1024
dropout = 0.25
learning_rate = 0.001
lr_decay = 0.95
channels = 1024

causal = False

viz_no_ground_truth = True  # do not show ground-truth poses
viz_bitrate = 3000
viz_limit = -1
viz_downsample = 1
viz_size = 5
viz_skip = 0
viz_fps = 30

resume = ''
evaluate = 'pretrained_h36m_detectron_coco.bin'
checkpoint = 'checkpoint'

dataset = CustomDataset(myvideos_path)

# print(dataset)
# print(dataset.cameras())
# print(dataset.fps())
# print(dataset.skeleton())
# print(dataset.subjects())


print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d


keypoints = np.load(myvideos_path, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
            action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps


# # print(keypoints)
# for k, v in keypoints.items():
#     print(k,v['custom'][0].shape)
#     # print(v['custom'][0][0][0])


# subjects_train = args.subjects_train.split(',')
# subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
# if not args.render:
#     subjects_test = args.subjects_test.split(',')
# else:
#     subjects_test = [args.viz_subject]
#
# semi_supervised = len(subjects_semi) > 0
# if semi_supervised and not dataset.supports_semi_supervised():
#     raise RuntimeError('Semi-supervised training is not implemented for this dataset')


#############################################################

print('Rendering...')

# print(keypoints)
input_keypoints = keypoints[viz_subject][viz_action][viz_camera].copy()
ground_truth = None
if viz_subject in dataset.subjects() and viz_action in dataset[viz_subject]:
    if 'positions_3d' in dataset[viz_subject][viz_action]:
        ground_truth = dataset[viz_subject][viz_action]['positions_3d'][viz_camera].copy()
if ground_truth is None:
    print('INFO: this action is unlabeled. Ground truth will not be rendered.')

######################################################

if not render:
    subjects_test = subjects_test.split(',')
else:
    subjects_test = [viz_subject]

semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError('Semi-supervised training is not implemented for this dataset')


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if actions == '*' else actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

filter_widths = [int(x) for x in architecture.split(',')]
if not disable_optimizations and not dense and stride == 1:
    # Use optimized model for single-frame predictions
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                                dense=dense)

model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                            filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                            dense=dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

if resume or evaluate:
    chk_filename = os.path.join(checkpoint, resume if resume else evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    print('''checkpoint : ''', checkpoint.keys())
    # print('''checkpoint['model_pos'] : ''', checkpoint['model_pos'])
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)


# ######################################################


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
                                                                                         inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev


########################################################

prediction = evaluate(gen, return_predictions=True)
# prediction = np.load(np_load_file_path)
# prediction = np.load('interpolation_test/RGB/ArmCrossNpy/a6_s1_t1_color.npy')



# ######################################################



if viz_export is not None:
    print('Exporting joint positions to', viz_export)
    # Predictions are in camera space
    np.save(viz_export, prediction)

if viz_output is not None:
    if ground_truth is not None:
        # Reapply trajectory
        trajectory = ground_truth[:, :1]
        ground_truth[:, 1:] += trajectory
        prediction += trajectory

    # Invert camera transformation
    cam = dataset.cameras()[viz_subject][viz_camera]
    if ground_truth is not None:
        prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
        ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
    else:
        # If the ground truth is not available, take the camera extrinsic params from a random subject.
        # They are almost the same, and anyway, we only need this for visualization purposes.
        for subject in dataset.cameras():
            if 'orientation' in dataset.cameras()[subject][viz_camera]:
                rot = dataset.cameras()[subject][viz_camera]['orientation']
                break
        prediction = camera_to_world(prediction, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    anim_output = {'Reconstruction': prediction}
    if ground_truth is not None and not viz_no_ground_truth:
        anim_output['Ground truth'] = ground_truth

    input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

    # render_animation(input_keypoints, keypoints_metadata, anim_output,
    #                  dataset.skeleton(), dataset.fps(), viz_bitrate, cam['azimuth'], viz_output,
    #                  limit=viz_limit, downsample=viz_downsample, size=viz_size,
    #                  input_video_path=viz_video, viewport=(cam['res_w'], cam['res_h']),
    #                  input_video_skip=viz_skip)

    render_animation(input_keypoints, keypoints_metadata, anim_output,
                     dataset.skeleton(), viz_fps, viz_bitrate, cam['azimuth'], viz_output,
                     limit=viz_limit, downsample=viz_downsample, size=viz_size,
                     input_video_path=viz_video, viewport=(cam['res_w'], cam['res_h']),
                     input_video_skip=viz_skip)









