import cv2
from yolo_predictions import YOLO_Pred
yolo=YOLO_Pred('Model/weights/best.onnx', 'data.yaml')
img=cv2.imread('test_2.jpg')
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
