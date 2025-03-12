# Torch Trainer 

<div align="center">
  <img src="./src/flare.svg" alt="" style="width: 120px"/>
  <p>Utilities for Training Pytorch</p>
  <p>내가 쓸려고 만든 pytorch 학습 도구</p>
</div>

---

I'm working on it...

- Train
  - validation
  - fp16 precision
  - TPU support
  - Cross validation
  - EarlyStopping
  - Warmup scheduler
  - Dynamic scheduler
- Inference
- Evaluation
- DataLoader
  - Custom dataloader
- Log
- Others
  - Fix random seed
  - Clear cache
  - Ignore warnings

Alternative: Pytorch-lightning + MLFlow

## Google Colab

Make sure to run this code before attempting to import the module.

```python
# Mount Google Drive for Colab env
import sys
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
sys.path.append("/content/drive/MyDrive")
```

### FP16 precision

Colab GPUs support FP16 precision, which significantly sped up training. (Specifically, 5 epochs took 207.32 seconds with standard precision to train the transformer and only 61.92 seconds with FP16.)
