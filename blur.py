import os 
import cv2
import dlib
 
def blur_face(img_path):
    file_path = os.path.join("static", img_path)
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        
        blurred = cv2.blur(img[top:bottom, left:right], (65, 65))
        img[top:bottom, left:right] = blurred

    cv2.imwrite(file_path, img)

    return img_path