# Handwriting-recognition-deep-learning-program-on-Raspberry-Pi
라즈베리파이 IoT : 직접 쓴 손글씨 인식 모델 
# 최종 결과물


https://github.com/user-attachments/assets/1c26fe0e-453b-4d80-bcba-8c1264cad6e3



## MNIST 글자 인식 딥러닝 모델 복습

[MNIST인식모델생성.ipynb](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/3e67e3f7-e8e7-477b-a547-d7f97b38b5ff/MNIST%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC.ipynb)

### **데이터 로드 및 전처리**

- **MNIST 데이터셋**은 `tensorflow.keras`의 내장 데이터셋에서 가져온다
- `x_train`과 `x_test` 데이터는 28x28 크기의 이미지로, 픽셀 값이 0에서 255 범위로 표현된다
- 데이터를 신경망에 적합하도록 **정규화**하여 픽셀 값을 0~1 사이로 변환한다

---

### **모델 정의**

- 딥러닝 프레임워크 TensorFlow/Keras를 사용하여 모델을 정의한다
- **Sequential API**를 통해 다음의 레이어 구조를 갖는 신경망을 구현한다
    - **Flatten Layer**: 입력 이미지를 1차원 벡터로 변환
    - **Dense Layer (512 units)**: 활성화 함수로 ReLU 사용
    - **Dropout Layer**: 과적합을 방지하기 위해 50%의 뉴런을 무작위로 비활성
    - **Dense Layer (10 units)**: 활성화 함수로 Softmax 사용하여 10개의 클래스(숫자 0~9)에 대한 확률 분포 출력

---

### **모델 컴파일**

- **손실 함수**: `sparse_categorical_crossentropy`를 사용하여 다중 클래스 분류 문제 해결
- **최적화 기법**: `adam` 옵티마이저 사용
- **평가지표**: `accuracy`로 모델 성능 평가

---

### **모델 훈련**

- `fit` 메서드를 사용하여 모델을 훈련시킨다
    - **훈련 데이터**: `x_train`과 `y_train` 사용
    - **배치 크기**: 32
    - **에포크 수**: 5
    - **검증 데이터**: `x_test`와 `y_test`를 사용하여 모델의 일반화 성능 확인

---

<img width="596" alt="1" src="https://github.com/user-attachments/assets/d43dc3bf-cf80-4a5d-b5b9-34268068e890" />


### **모델 평가**

- `evaluate` 메서드를 통해 테스트 데이터(`x_test`)에서 모델의 성능을 확인한다
- 정확도(accuracy)는 모델이 테스트 데이터에서 얼마나 잘 작동하는지를 나타낸다

---

<img width="439" alt="2" src="https://github.com/user-attachments/assets/c8387c1c-4e8a-4162-a5cc-9ac98e4d72ff" />



### **시각화**

- MNIST 데이터셋의 이미지를 그리드(grid) 형태로 시각화한다
- 선택된 숫자 이미지를 표시하고, 모델의 예측 결과와 비교한다

## 직접 쓴 손글씨 인식 모델 만들기

[손글씨인식_모델활용.ipynb](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/8911c483-9c16-4cef-8bf9-3929254f64cf/%E1%84%89%E1%85%A9%E1%86%AB%E1%84%80%E1%85%B3%E1%86%AF%E1%84%8A%E1%85%B5%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8_%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%E1%84%92%E1%85%AA%E1%86%AF%E1%84%8B%E1%85%AD%E1%86%BC.ipynb)

<img width="547" alt="4" src="https://github.com/user-attachments/assets/3d384cbc-ec39-4c21-aae0-aa87df42b044" />
<img width="486" alt="3" src="https://github.com/user-attachments/assets/bd0301db-d026-4963-87cb-d3e976f1f799" />
<img width="550" alt="5" src="https://github.com/user-attachments/assets/d6016b85-f940-468b-9994-8d9407d52485" />

### **손글씨 이미지 로드 및 전처리**

- 손글씨 이미지를 불러와서 모델이 처리 가능한 형태로 변환한다
    - 이미지 크기를 **28x28 픽셀**로 변환한다
    - **그레이스케일로 변환**하여 흰색 배경과 검은색 글씨를 반전(255에서 빼기)한다
    - 각 픽셀 값을 **정규화(0~1)** 하여 신경망 입력에 적합하게 만든다

---

### **딥러닝 모델 불러오기**

- 사전에 학습된 Keras 모델(`digits_model.h5`)을 불러온다
- 모델의 구조는 다음과 같다:
    - **Flatten Layer**: 입력 이미지를 1차원 벡터로 변환
    - **Dense Layer (128 units)**: 활성화 함수 ReLU 사용
    - **Dropout Layer**: 과적합 방지를 위해 일부 뉴런 비활성화
    - **Dense Layer (10 units)**: 활성화 함수 Softmax 사용, 숫자 0~9 분류
- 모델 파라미터 수: 약 **101,770개**

---

<img width="563" alt="6" src="https://github.com/user-attachments/assets/dae74e25-07ed-446c-b6d2-c1d6721572bc" />


### **이미지 예측**

- 모델을 사용하여 각 손글씨 이미지에 대해 예측을 수행한다
    - `model.predict` 메서드를 사용하여 이미지의 숫자 클래스를 예측한다
    - 예측 결과에서 가장 높은 확률을 가진 클래스를 `argmax`로 출력한다
    - 각 숫자에 대한 확률 분포도 함께 출력한다

---

<img width="634" alt="7" src="https://github.com/user-attachments/assets/483d11cf-9b2f-445d-9569-43be4c421404" />


### **시각화**

- 원본 이미지를 시각화하고, 예측된 숫자와 각 클래스 확률을 표시한다
- 손글씨 숫자를 사람이 직접 확인하면서 모델의 성능을 평가할 수 있다

## 웹카메라 손글씨 영상 인식 딥러닝 프로그램

[손글씨인식_웹카메라_프로그램.py](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/34264a2b-bede-44f3-b1fe-a5f03d49fee1/%E1%84%89%E1%85%A9%E1%86%AB%E1%84%80%E1%85%B3%E1%86%AF%E1%84%8A%E1%85%B5%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8_%E1%84%8B%E1%85%B0%E1%86%B8%E1%84%8F%E1%85%A1%E1%84%86%E1%85%A6%E1%84%85%E1%85%A1_%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7.py)

**웹캠을 이용하여 실시간으로 손글씨 숫자를 인식하는 프로그램**



https://github.com/user-attachments/assets/7b5e9077-2b6d-49a0-b6d0-372b0e662306



### **주요 라이브러리**

- **OpenCV (`cv2`)**: 웹캠 영상 처리 및 컴퓨터 비전 작업
- **TensorFlow (`tf`)**: 사전에 학습된 딥러닝 모델 불러오기 및 예측 수행
- **Tkinter (`tkinter`)**: 간단한 GUI(그래픽 사용자 인터페이스) 생성
- **Pillow (`PIL`)**: 이미지를 Tkinter에서 표시하기 위해 변환

---

### **주요 설정**

- **기본 모델**: `digits_model.h5`를 불러온다 (사전에 학습된 MNIST 손글씨 숫자 인식 모델)
- **화면 크기**: 웹캠 프레임은 **300x300 픽셀**로 리사이즈된다
- **영상 소스**: `default_src` 변수로 웹캠 디바이스를 선택 (기본값은 0)

---

### **주요 함수**

1. **`start()`**:
    - 웹캠 스트림을 초기화하고, 영상 소스를 설정
    - `cv2.VideoCapture`를 사용해 실시간 영상을 읽어들인다
    - 스트림이 준비되면 `detectAndDisplay` 함수를 호출하여 처리한다
2. **`detectAndDisplay()`**:
    - 웹캠으로부터 프레임을 가져온다
    - 프레임의 크기를 **300x300**으로 조정한다
    - **HSV 변환**을 통해 밝기(`value`) 채널을 추출한다
    - 사전 학습된 모델을 이용하여 숫자를 예측한다
    - **숫자 인식 결과를 화면에 표시한다**

---


<img width="2489" alt="8" src="https://github.com/user-attachments/assets/eaa40cc4-7649-40d9-b1ba-9cc2f66f2d52" />

### **주요 워크플로**

1. **숫자 전처리**:
    - 추출한 밝기(`value`) 이미지를 흑백 반전시키고, MNIST 입력 형식(28x28)으로 변환한다
    - 픽셀 값을 0~1 사이로 정규화한다
2. **모델 예측**:
    - 모델의 `predict` 메서드를 사용하여 각 숫자의 클래스 확률을 계산한다
    - 가장 높은 확률을 가진 숫자를 인식 결과로 표시한다
3. **GUI 표시**:
    - Tkinter와 Pillow를 활용하여 웹캠 프레임 및 예측된 결과를 GUI 창에 표시한다

# 라즈베리파이 직접 쓴 손글씨 인식하기



https://github.com/user-attachments/assets/52e25a55-b972-401f-a92b-a8ce877addb6



- **cv2**: OpenCV 라이브러리로 비디오 및 이미지 처리에 사용됨
- **numpy**: 수치 연산과 행렬 계산을 위한 라이브러리
- **tensorflow**: 사전 학습된 **Keras 모델**을 불러와 숫자를 예측하는 데 사용됨

```kotlin
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('digits_model.h5')

# Constants
SZ = 28
margin = 30
frame_width = 300
frame_height = 300

# Initialize the video capture (0 for webcam or replace with file path)
video_path = "path_to_video.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot fetch the frame.")
        break

    # Resize frame to fixed dimensions
    frame = cv2.resize(frame, (frame_width, frame_height))

    # HSV transform - value = gray image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, value = cv2.split(hsv)

    # Kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply topHat and blackHat operations
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

    # Add and subtract operations
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small sections
        if w * h < 2400:
            continue

        y_position = max(y - margin, 0)
        x_position = max(x - margin, 0)
        img_roi = thresh[y_position:y + h + margin, x_position:x + w + margin]

        # Resize ROI to model input size
        num = cv2.resize(img_roi, (SZ, SZ))
        num = num.astype('float32') / 255.0

        # Predict the number
        result = model.predict(np.array([num]))
        result_number = np.argmax(result)

        # Draw bounding box and result
        cv2.rectangle(frame, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 255, 0), 2)
        text = "Number: {}".format(result_number)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow("MNIST Hand Write", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

```

- 사전 학습된 MNIST 숫자 인식 모델을 `digits_model.h5` 파일에서 로드함
- 해당 모델은 입력받은 숫자 이미지를 기반으로 **0부터 9까지의 숫자**를 예측하는 신경망이다
- `SZ`: 모델 입력 이미지의 크기 (28x28 픽셀)로 MNIST 데이터셋의 표준 크기
- `margin`: 숫자 주변에 추가적으로 잡히는 여백의 크기
- `frame_width`, `frame_height`: 비디오 프레임의 가로, 세로 크기를 고정시킴
- 비디오 파일 경로를 `video_path` 변수에 설정
- `cv2.VideoCapture`를 통해 비디오를 읽어옴
- 웹캠을 사용하려면 `"path_to_video.mp4"` 대신 `0`을 입력하면 된다
- `cap.read()`를 사용해 비디오의 프레임을 읽어옴
- 프레임의 크기를 `frame_width`와 `frame_height`로 조정
- **HSV** 색상 공간으로 변환하여 **Value (명도)** 채널만 추출
- 명도 채널을 사용하는 이유는 조명 변화에 강하고 흑백 이미지에 집중하기 위
- **kernel**: (3x3) 사각형 커널
- **TopHat 연산**: 원본 이미지에서 열림(Opening) 연산 결과를 빼서 밝은 부분 강조
- **BlackHat 연산**: 닫힘(Closing) 연산 결과에서 원본 이미지를 빼서 어두운 부분 강조
- **TopHat**과 **BlackHat** 연산으로 강조된 명도를 더하고 빼서 대비를 높임
- **Gaussian Blur**로 노이즈를 제거하고 부드럽게 만듦
- **Adaptive Thresholding**: 이미지의 조명 변화에 대응하여 국소적으로 임계값을 적용
- `THRESH_BINARY_INV`: 흰 배경, 검은 물체를 반전시켜 숫자가 흰색으로 보이게 설정
- **findContours**: 이진화된 이미지에서 숫자의 윤곽선을 검출
- **RETR_LIST**: 모든 윤곽선을 계층 없이 검출
- **CHAIN_APPROX_SIMPLE**: 윤곽선을 단순화하여 저장
- `boundingRect`: 윤곽선을 감싸는 직사각형 좌표 `(x, y, w, h)`를 구함
- 작은 객체(노이즈)를 무시하기 위해 면적이 `2400` 미만인 객체는 건너뜀
- 숫자 주변에 **margin** 만큼의 여백을 추가하고 ROI(Region of Interest)를 추출
- ROI를 모델 입력 크기인 `28x28`로 리사이즈
- 0~1 범위로 정규화
- 모델에 입력하고 결과에서 가장 높은 확률을 가진 숫자를 예측함
- 숫자 주변에 **초록색 박스**를 그림
- 예측된 숫자를 프레임 위에 텍스트로 출력

프레임을 실시간으로 화면에 보여주고 `q` 키를 누르면 종료
