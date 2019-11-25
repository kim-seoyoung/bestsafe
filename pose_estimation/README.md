# 3D pose estimation
### VNect
* [github](https://github.com/XinArkh/VNect)
* (모델 설명 쓰기)

#### 설치

```bash
~$ git clone https://github.com/XinArkh/VNect
~$ cd VNect
```

#### 사용법
```bash
VNect~$ python3 run_estimator_ps.py --input {input_file} --output-dir {output_directory}
```
- 3D plot을 gif로 저장
```bash
VNect~$ python3 run_estimator_ps.py --input {input_file} --output-dir {output_directory} --savegif True
```


- 도커 이미지 올리기
