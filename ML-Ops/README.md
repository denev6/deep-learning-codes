# ML API

- Dependencies: [conda.yaml](/ML-Ops/_model/conda.yaml)
- Config: [MLproject](/ML-Ops/_model/MLproject)
- Train model using Torch-Vision: [final.ipynb](/ML-Ops/final.ipynb)
  - FashionMNIST
  - MobileNet_v2
- Trace model on MLFlow: [model](/ML-Ops/_model)
- Serve model using FastAPI: [main.py](/ML-Ops/app/main.py)
- Docker settings: [Dockerfile](/ML-Ops/app/Dockerfile)

![API preview](https://denev6.github.io/assets/posts/ml-api/prediction-img.png)

## 🌊 MLFlow

### Open MLFlow-ui

```bash
cd _model
mlflow ui
```

### Train

```bash
mlflow run . -e main --experiment-name=MobileNetV2 -P epoch=3 -P lr=0.005
```

### Predict

```bash
mlflow run . -e predict -P img="sample.png"
```

## 📗 FastAPI

```bash
cd app
uvicorn main:app --reload
```

### Test API

```bash
python test.py
```

## 🐋 Docker

```bash
cd app
docker build -t app:0.1 .
docker image ls -a
docker run -p 8080:8000 --name test app:0.1
```

URL: 127.0.0.1:8080

### Remove

```bash
docker rm test
docker rmi app:0.1
```
