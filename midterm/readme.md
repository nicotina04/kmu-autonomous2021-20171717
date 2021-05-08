# 차량지능기초 전반부 프로젝트
## 프로젝트 수행자
20171717 소프트웨어학부 최호경

## 프로젝트 주제
폭력 사건 인지를 위한 인간의 일상, 사격 자세 분류

[발표 동영상](https://studio.youtube.com/video/cRq21xyuajw/edit)

## Project environment
python: 3.8.2

opencv-python: 4.5.1

sklearn: 0.24.2

* * *

# Summary of files

## dataCreator.py
처음 사람의 포즈를 인식하여 데이터셋 포맷을 초기화 하고 동영상에 클래스를 부여하여 데이터셋을 생성하는 코드

## gunnerDetector.pkl
Pretrained model for project

## poseModel.py
csv 형태로 수집한 데이터 셋을 SVM으로 훈련하는 코드

## solution.py
Pretrained model을 가지고 웹캠을 통해 직접 pose를 분류하는 코드

* * *

# How to use

## 당장 사용하길 원한다면
pretrained model이 gunnerDetector.pkl로 제공됩니다. 카메라가 있는 컴퓨터에서 solution.py를 실행해 자세 분류를 할 수 있습니다.

## 처음부터 해보기
카메라가 있는 컴퓨터 환경에서 실행해야 합니다.

먼저 dataCreator.py를 실행한 뒤 1을 입력하면 캠으로 사람의 포즈에 대해 어노테이션을 한 영상이 출력됩니다.

esc키를 눌러 카메라를 종료하면 데이터 포맷이 초기화되어 csv파일로 저장됩니다.

그 다음 데이터를 추가하기 위해 클래스 이름으로 된 폴더를 생성하고 폴더 안에 해당 클래스로 레이블할 동영상을 넣습니다.

다시 dataCreator.py를 실행하고 2번을 입력한 뒤 해당 클래스 이름을 입력하면 opencv에서 동영상을 읽고 mediapipe의 pose solution으로 인식한 결과가 csv 파일에 추가됩니다.

데이터셋 구성이 끝났으면 poseModel.py를 실행해 학습을 수행할 수 있습니다.

학습이 끝난 모델을 solution.py을 실행하여 사용합니다.
