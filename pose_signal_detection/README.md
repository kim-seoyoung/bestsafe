# Signal Detection

## model
### VideoPose3D
* ([link](https://github.com/facebookresearch/VideoPose3D))
* 수신호로 사용할 dataset을 videopose3d를 이용해서 skeleton을 추출하고 영상에 rendering
![a2_s8_t2_color_converted](https://user-images.githubusercontent.com/52961246/68527851-16422000-032f-11ea-9a1e-59fe3bb3e565.gif)

### VNect

## Dataset
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


