from PIL import Image
from torchvision import transforms
import onnx
import onnxruntime

onnx_path = "../../models/cnn.onnx"
img_path = "../../assets/digit.png"

# Check model status
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Read a sample image
image = Image.open(img_path)
image = image.convert("L")
preprocess = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
input_tensor = preprocess(image)
numpy_array = input_tensor.numpy().reshape(1, 1, 28, 28)

# Run model on ONNX
session = onnxruntime.InferenceSession(onnx_path)
 
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
 
outputs = session.run([output_name], {input_name: numpy_array})
prediction = outputs[0].squeeze(0).argmax(-1)
assert prediction == 3

print("No Problem!")
