from label_detector import label_detector
from ocr import ocr
import cv2

img_path = 'ex16.jpg'
extractor = label_detector(img_path)
label_image = extractor.get_final_image()

cv2.imwrite('test_label.jpg',label_image)

ocr(label_image)