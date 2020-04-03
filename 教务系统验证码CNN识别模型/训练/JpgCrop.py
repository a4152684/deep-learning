#预处理教务系统验证码原始图片，得到分割后的结果
import os
from PIL import Image
from JpgPretreat import twoValue,clearNoise,saveImage,cfs,saveSmall

CAPTCHA_IMAGE_FOLDER = "../爬取验证码/captcha_images"
extracted_letter_images="split_images"

for file in os.listdir(CAPTCHA_IMAGE_FOLDER):
    print(file)
    #image = Image.open('./jpg/3.jpg').convert("L")
    image = Image.open('./'+CAPTCHA_IMAGE_FOLDER+'/'+file).convert("L")#PIL库中Image类的将RGB图像转换为灰度图像的方法
    twoValue(image, 100)#设置阈值100，并二值化
    clearNoise(image, 1, 1)#去除噪声
    image=saveImage(image.size)#将原图片用已经去噪和二值化的图片覆盖
    x,y=cfs(image)#二值化图像分割
    saveSmall(extracted_letter_images, file, image, x,y)#保存分割后的图片

print('end')



