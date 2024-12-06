import os 
from PIL import Image, ImageDraw
import cv2  
import re



def blackout_image(img_path, cam_pos):
    # save_path = re.sub(img_root, save_root, img_path)
    # save_dir = os.path.dirname(save_path)
    # os.makedirs(save_dir, exist_ok=True)
    # basename = os.path.basename(img_path)
    img = Image.open(img_path)
    # cam_id = basename.split(".")[0].split("_")[int(cam_pos)]
    # print(save_path)
    # print(cam_id)

    if cam_pos == "0":
        x1, y1 = 206, 250
        x2, y2 = 125,250
        x3, y3 = 15,324  
        x4, y4 = 324, 324
    elif cam_pos == "1":
        x1, y1 = 0, 200
        x2, y2 = 193, 178
        x3, y3 = 324, 178
        x4, y4 = 324, 324
        x5, y5 = 0, 324
    elif cam_pos == "2":
        x1, y1 = 0, 166
        x2, y2 = 135, 166
        x3, y3 = 324, 190
        x4, y4 = 324, 324
        x5, y5 = 0, 324

    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5,y5)], fill=255)

    blacked_out_img = Image.composite(Image.new('RGB', img.size, (0, 0, 0)), img, mask)
    # blacked_out_img.save(save_path)
    blacked_out_img.show()

cam_0 = "/home/walter/Downloads/imagr-031123_imagr-new-cam-angles_9310072032771_0_1698987847079_13675968.jpg"
cam_1 = "/home/walter/Downloads/imagr-031123_imagr-new-cam-angles_9300617074434_1_1698987198266_13034244.jpg"
cam_2 = "/home/walter/Downloads/imagr-031123_imagr-new-cam-angles_9310072032771_2_1698987813251_13657889.jpg"

# cam0 = cv2.imread(cam_0)
# cam1 = cv2.imread(cam_1)
# cam2 = cv2.imread(cam_2)

# cv2.imshow("cam1", cam2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
blackout_image(cam_2, "2")

# cam_0 = Image.open(cam_0)
# cam_0.show()