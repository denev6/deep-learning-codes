import mlflow

logged_model = "runs:/ffb263113ee64d05928679ef5bcda173/MobileNet_model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

#######################################################################
# Predict on a Numpy Array.
from PIL import Image

from mobilenet import transform, labels

image_path = "./data/sample.png"

image = Image.open(image_path)
input_tensors = transform(image)
input_tensor = input_tensors.unsqueeze(0)
input_array = input_tensor.cpu().numpy()

output = loaded_model.predict(input_array)
predicted = output.argmax()
print(f"Predicted: {labels[predicted.item()]}")
# Predicted: Ankle boot
