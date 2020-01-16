# Signal Detection

![process](https://user-images.githubusercontent.com/39910353/72503323-c75cbd80-387e-11ea-8772-48f7b3af9e96.PNG)

## step1: Dataset
* **UTD Multimodal Human Action Dataset** ([link](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html))
  - Microsoft Kinect sensor와 inertial sensor를 각각 하나씩 이용
  - 8명의 사람의 27가지의 action으로 구성
  - RGB videos, depth videos, skeleton joint positions, inertial sensor signal이 포함되어 있음
  ![example_images](https://user-images.githubusercontent.com/52961246/67183193-dcf74e00-f41b-11e9-924e-9c66ff348eb9.png)
  - Draw X, Draw circle 등 실제 사용 수신호와 유사한 data이용 
  - Kinect v2이용한 추가적인 dataset존재


## Signal 참고
* **ISO 16715 크레인 안전 수신호**  
STOP, EMERGENCY STOP 신호가 쓸만해보임  
<img src="https://user-images.githubusercontent.com/54068348/67183194-dcf74e00-f41b-11e9-9a53-47ed2c752b0e.png" width="50%"/>

* **덤프트럭 수신호**  
HOLD, RAISE/LOWER, LOOK, SLOW DOWN, SPEED UP 신호가 쓸만해보임  
<img src="https://user-images.githubusercontent.com/54068348/67183625-e92fdb00-f41c-11e9-98fb-dfa06002cd64.png" width="50%"/>


## step2: model

* 모델 구축은 VideoPose3D,VNect 두가지 모델을 사용할 수 있고 각각 두개의 버전으로 Signal Detection Model을 구축해 볼 수 있다. 

### VideoPose3D model
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* 동영상의 각 프레임으로 부터 2D 관절을 먼저 추출한 후 2D관절의 연속성에 따라 3D 관절을 추출해 내는 모델로 17개 관절을 추출한다.
* 2D 관절을 추출하고 3D 관절을 추출하는 2가지 과정을 거친다. 이때 2D 관절은 'Detectron'모델을 사용한다
* [Detectron](https://github.com/facebookresearch/Detectron)

* ffmpeg를 통한 .mp4로 변환
  -> detectron을 통한 2d keypoint추출(.npz) -> videopose3d를 통해 2d keypoint를 3d skeleton으로 변환
<img src="https://user-images.githubusercontent.com/52961246/68527903-c6b02400-032f-11ea-9384-bb9bbbc32d34.png" width="300"/>

* 수신호로 사용할 dataset을 videopose3d를 이용해서 skeleton을 추출하고 영상에 rendering
![a2_s8_t2_color_converted](https://user-images.githubusercontent.com/52961246/68527851-16422000-032f-11ea-9a1e-59fe3bb3e565.gif)

### VNect model

* [VNect_pose_estimaion](https://github.com/kim-seoyoung/bestsafe/tree/master/pose_estimation)


## step3: Dataset 전처리

[2s-AGCN](https://github.com/lshiwjx/2s-AGCN) 모델을 참고하였다.  

### Data format transfer

dataset을 전처리 하기 위해 **data/mydata/npy**에 skeleton을 추출한 npy파일을 넣은 후 아래 명령어를 실행한다.

`python data_gen/my_gendata.py`

위의 명령어를 실행하면 **data/mydata/gen**에 결과물이 생성될 것이다. 이 결과물은 데이터를 트레이닝하기 위해 모델의 입력부분에 알맞게 수정한 데이터이다.

Ex) 데이터의 shape 변경 : (60, 17, 3) -> (3, 100, 17, 2)


### Interpolation

spline interpolation 기법을 이용하여 






## step4: Classification


