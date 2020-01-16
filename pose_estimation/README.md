# 3D pose estimation

## VNect 모델
* [github](https://github.com/XinArkh/VNect)
* 영상의 각 프레임들로부터 2D, 3D joint heatmap을 추출하여 3D skeleton을 계산하고 21개의 관절을 추출해내는 모델이다.

## interpolation
* 분류 모델에 사용하기 위해 frame 수를 100으로 맞추기 위해 사용
* (설명)

#### Environments
- python 3.5
  - opencv-python 3.4.4.19
  - tensorflow-gpu 1.12.0 (CUDA 9.0)
  - pycaffe
  - matplotlib 3.0.0 or 3.0.2
  - imagemagick (apt-get)
 
#### Setup
```bash
~$ git clone https://github.com/kim-seoyoung/bestsafe
~$ cd bestsafe/pose_estimation
```

#### Usage
###### - numpy파일로 저장
```bash
~$ python3 run_estimator_ps.py --input {input_file} --output-dir {output_directory}
```
###### - numpy파일로 저장 + 3D plot을 gif로 저장
```bash
~$ python3 run_estimator_ps.py --input {input_file} --output-dir {output_directory} --savegif True
```


- (도커 이미지 올리기)


## VideoPose3D 모델
* [github](https://github.com/facebookresearch/VideoPose3D)
* 동영상의 각 프레임으로 부터 2D 관절을 먼저 추출한 후 2D관절의 연속성에 따라 3D 관절을 추출해 내는 모델로 17개 관절을 추출한다.
* 2D 관절을 추출하고 3D 관절을 추출하는 2가지 과정을 거친다. 이때 2D 관절은 'Detectron'모델을 사용한다
* [Detectron](https://github.com/facebookresearch/Detectron)
