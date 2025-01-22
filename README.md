# 딥러닝 프로젝트

- [RoBERTa Fine-tuning](#roberta-fine-tuning)
- [OpenCV 이미지 처리](#opencv-이미지-처리)
- [ML 모델 서빙](#ml-모델-서빙)

## RoBERTa Fine-tuning

`Python` `Hugging-Face` `Torch` `BERT` `Data-processing`

README: [mobilenet mlflow](/roberta)

- Model: [Github](/roberta/RoBERTa_pytorch.ipynb) | [DACON](https://dacon.io/competitions/official/236027/codeshare/7275)

<img src="https://denev6.github.io/assets/posts/dacon-2022/award.png" alt="Ranked in 2/259" style="max-width:300px">

## OpenCV 이미지 처리

`C++` `Python` `OpenCV` `Transformation` `Detection` `CNN`

README: [OpencvCpp](/OpencvCpp)

- 참고 도서: [OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝](https://sunkyoo.github.io/opencv4cvml/)
- Transformation: [C++](/OpencvCpp/src/geometry/transform.cpp) | [BLOG](https://denev6.github.io/computer-vision/2025/01/03/transformation.html)
- Edge detection: [C++](/OpencvCpp/src/geometry/edge.cpp) | [BLOG](https://denev6.github.io/computer-vision/2025/01/06/edge-detection.html)
- Object detection: [C++](/OpencvCpp/src/geometry/detection.cpp)
- Deep learning: [Run](/OpencvCpp/src/machine-learning/cnn_mnist.cpp) | [Train](/OpencvCpp/src/machine-learning/cnn_onnx.ipynb)

![edge detection](https://denev6.github.io/assets/posts/edge-detection/canny-result.png)

## ML 모델 관리 및 배포

`Python` `MLFlow` `Torch-vision` `FastAPI` `Docker` `MobileNet` `FashionMNIST`

README: [ML-Ops](/ML-Ops)

- MLFlow: [MLproject](/ML-Ops/_model/MLproject)
- Model: [final.ipynb](/ML-Ops/final.ipynb)
- FastAPI: [app](/ML-Ops/app)
- Docker: [Dockerfile](/ML-Ops/app/Dockerfile)

![mlflow preview](/ML-Ops/preview/prediction.png)
