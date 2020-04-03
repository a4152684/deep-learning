#预处理教务系统验证码原始图片的库
#原图片->灰度化->二值化和降噪->字符切割并归一化
#import sys
from PIL import Image, ImageDraw
import queue
#import os,time
#from time import sleep
#import datetime
#import matplotlib.pyplot as plt
import cv2
#import pandas as pd

a=[0,1,2,3]
t2val = {}
def cfs(img):#img此时已经经过去噪，二值化
    """传入二值化后的图片进行连通域分割"""
    pixdata = img.load()#方法load()返回一个用于读取和修改像素的像素访问对象,这个访问对象像一个二维队列
    w,h = img.size
    visited = set() #创建一个集合
    q = queue.Queue() #单向队列，先进先出
    offset = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    x_cuts = []
    y_cuts = []
    for x in range(w):
        for y in range(h):
            x_axis = []
            y_axis = []
            if pixdata[x,y] == 0 and (x,y) not in visited:
                q.put((x,y))#如果(x,y)处像素值为0且其没有被访问过(即不在集合visited中)
                visited.add((x,y))#则将其放入队列q，并且设置为访问过
            while not q.empty():#当队列不空的时候
                x_p,y_p = q.get()#取出最先进入的一个
                for x_offset,y_offset in offset:
                    x_c,y_c = x_p+x_offset,y_p+y_offset
                    if (x_c,y_c) in visited:
                        continue
                    visited.add((x_c,y_c))
                    try:
                        if pixdata[x_c,y_c] == 0:
                            q.put((x_c,y_c))
                            x_axis.append(x_c)
                            y_axis.append(y_c)
                            #y_axis.append(y_c)
                    except:
                        pass
            if x_axis:
                min_x,max_x = min(x_axis),max(x_axis)
                min_y,max_y = min(y_axis),max(y_axis)
                if max_x - min_x >  3:
                    # 宽度小于3的认为是噪点，根据需要修改
                    x_cuts.append((min_x,max_x + 1))
                    y_cuts.append((min_y,max_y + 1))
    return x_cuts,y_cuts





def clearNoise(image, N, Z):
    for i in range(0, Z):
        t2val[(0, 0)] = 1
        t2val[(image.size[0] - 1, image.size[1] - 1)] = 1

        for x in range(1, image.size[0] - 1):
            for y in range(1, image.size[1] - 1):
                nearDots = 0
                L = t2val[(x, y)]
                if L == t2val[(x - 1, y - 1)]:
                    nearDots += 1
                if L == t2val[(x - 1, y)]:
                    nearDots += 1
                if L == t2val[(x - 1, y + 1)]:
                    nearDots += 1
                if L == t2val[(x, y - 1)]:
                    nearDots += 1
                if L == t2val[(x, y + 1)]:
                    nearDots += 1
                if L == t2val[(x + 1, y - 1)]:
                    nearDots += 1
                if L == t2val[(x + 1, y)]:
                    nearDots += 1
                if L == t2val[(x + 1, y + 1)]:
                    nearDots += 1

                if nearDots < N:
                    t2val[(x, y)] = 1#若邻域内与其相等的像素个数很少，则认为是空白点



def saveSmall(SaveDir,file, img, x_cuts,y_cuts):#将分割后的图片保存
    w, h = img.size
    #pixdata = img.load()
    file = file.split('.')[0]#取文件名而忽视后缀
    #print(file)
    for j, item in enumerate(x_cuts): #j为列表索引，item为j对应的值
        if j<4:
            box = (item[0], 0, item[1], 30)#(left, upper, right, lower)，本图片中字母高度不变
            a[j]=img.crop(box)
            # crop() : 从图像中提取出某个矩形大小的图像。它接收一个四元素的元组作为参数，各元素为（left, upper, right, lower），坐标系统的原点（0, 0）是左上角。
            a[j].save('./'+SaveDir+'/'+file+'_'+str(j)+'.png')
            #print(box)
            #img.crop(box).save(outDir+file[j]+'/' + str(i)+'-'+str(j) + ".png")

def twoValue(image, G):#一般比clear_noise先运行
    for y in range(0, image.size[1]):
        for x in range(0, image.size[0]):
            g = image.getpixel((x, y))
            if g > G:
                t2val[(x, y)] = 1
            else:
                t2val[(x, y)] = 0#小于阈值则视作黑色点


def saveImage(size):
    image = Image.new("1", size) #创建一个模式为“1”，尺寸为size的新图像
    draw = ImageDraw.Draw(image)#创建一个可以在给定图像image上绘图的对象draw，在draw上操作，image会被修改

    for x in range(0, size[0]):
        for y in range(0, size[1]):
            draw.point((x, y), t2val[(x, y)])

    return image

# rename and convert to 30*30 size
def convert(dir, file):

    imagepath = dir+'/'+file
    # 读取图片
    image = cv2.imread(imagepath, 0)
    # 二值化
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    img = cv2.resize(thresh, (30, 30), interpolation=cv2.INTER_AREA)
    # 保存图片
    cv2.imwrite('%s/%s' % (dir, file), img)

# 读取图片的数据，并转化为0-1值
def Read_Data(dir, file):

    imagepath = dir+'/'+file
    # 读取图片
    image = cv2.imread(imagepath, 0)
    # 二值化
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # 显示图片
    bin_values = [1 if pixel==255 else 0 for pixel in thresh.ravel()]

    return bin_values

def img2list(image):
    twoValue(image, 100)#设置阈值100，并二值化
    clearNoise(image, 1, 1)#去除噪声
    image=saveImage(image.size)#将原图片用已经去噪和二值化的图片覆盖
    x_cuts,y_cuts=cfs(image)#二值化图像分割
    image_list=[]
    
    for j, item in enumerate(x_cuts): #j为列表索引，item为j对应的值
        if j<4:
            box = (item[0], 0, item[1], 30)#(left, upper, right, lower)，本图片中字母高度不变
            image_list.append(image.crop(box))
            
    return image_list



