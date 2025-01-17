import requests

# Test API
port = "8080"
url = f"http://127.0.0.1:{port}/predict/"
image_path = "./static/996.jpg" # sample image

with open(image_path, "rb") as image_file:
    # { Field-name: File-name, File-object, File-type }
    files = {"file": (image_path, image_file, "image/jpeg")}
    response = requests.post(url, files=files)
    resp_json = response.json()

print("Status:", response.status_code)
print("Response:", resp_json)

assert response.status_code == 200
assert resp_json["label"] == "Sneaker"
assert isinstance(resp_json["prob"], (float, int)) is True
print("=== Test Completed! ===")
