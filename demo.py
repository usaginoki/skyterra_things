from PIL import Image
from src_demo import ExperimentalClassifier
import os

img_path_0 = "../data/folder1/Almaty/Enbekshikazakhski/{2DE323DD-3C9D-48D3-9F7F-927C740F38FB}/do_poseva/20240325_045752_05_24a8_3B_Visual_clip.tif"
img_path_1 = "../data/folder1/Almaty/Enbekshikazakhski/{2DE323DD-3C9D-48D3-9F7F-927C740F38FB}/posev/20240402_055236_04_24fb_3B_Visual_clip.tif"

img_0 = Image.open(img_path_0)
img_1 = Image.open(img_path_1)

classifier = ExperimentalClassifier()

class_0_path = "../data/parsed/class_0"
class_1_path = "../data/parsed/class_1"


# class 0 demo
img_path = "../data/parsed/class_0/{DE7AE357-1816-45E6-9C11-4EFD025C8581}_17.04.2024_psscene_visual_20240417_044335_27_24bc_3B_Visual_clip_class_0_0637.tif"

image = Image.open(img_path)
print(classifier.classify_image(img_path))


# class 1 demo
img_path = "../data/parsed/class_1/{3FB9501A-671E-49AA-AE7E-3050F2AE7288}_23.04.2024_psscene_visual_20240423_053407_91_2498_3B_Visual_clip_class_1_0598.tif"

image = Image.open(img_path)
print(classifier.classify_image(img_path))


# unidentified class demo
unidentified_img_path = "../data/parsed/unidentified/PSScene_20240422_060632_28_242b_3B_Visual_clip_unidentified_0026.tif"

image = Image.open(unidentified_img_path)
print(classifier.classify_image(unidentified_img_path))


# class 0 failed demo
# find a failed class 0 image
class_0_path = "../data/parsed/class_0"
for img_path in os.listdir(class_0_path):
    img_path = os.path.join(class_0_path, img_path)
    if classifier.classify_image(img_path) == 1:
        print(img_path)
        failed_img_path = img_path
        break

failed_img_path = "../data/parsed/class_0/{8773D12F-FC86-427E-8710-B0C7E060A4D8}_4.05.2024_psscene_visual_20240504_053649_05_247b_3B_Visual_clip_class_0_0576.tif"

image = Image.open(failed_img_path)
gray_image = image.convert("L")
gray_image.show()

# class 1 failed demo
# find a failed class 1 image
class_1_path = "../data/parsed/class_1"
for img_path in os.listdir(class_1_path):
    img_path = os.path.join(class_1_path, img_path)
    if classifier.classify_image(img_path) == 0:
        print(img_path)
        failed_img_path = img_path
        break

failed_img_path = "../data/parsed/class_1/posev_20240517_044018_53_24c2_3B_Visual_clip_class_1_0123.tif"

image = Image.open(failed_img_path)
