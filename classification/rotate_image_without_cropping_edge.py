import cv2
import math

def rotate_image(array, angle):
   height, width = array.shape[:2]
   image_center = (width / 2, height / 2)
   
   rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
   
   radians = math.radians(angle)
   sin = math.sin(radians)
   cos = math.cos(radians)
   bound_w = int((height * abs(sin)) + (width * abs(cos)))
   bound_h = int((height * abs(cos)) + (width * abs(sin)))
   
   rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
   rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])
   
   rotated_mat = cv2.warpAffine(array, rotation_mat, (bound_w, bound_h))
   return rotated_mat


img = cv2.imread('/home/walter/nas_cv/walter_stuff/raw_data/crops/029470000105/0/imagr-051023_dylan_029470000105_0_1696479096592_471587.jpg',1)
print(img.shape)
rotated_image = rotate_image(img, 45)
print(rotated_image.shape)


cv2.imshow('Rotated image', rotated_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()