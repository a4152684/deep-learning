#抓取教务系统验证码原始图片
import requests
import os
import os.path
import time

CAPTCHA_IMAGE_FOLDER = "captcha_images"

img_url = 'http://bkjw.whu.edu.cn/servlet/GenImg'

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36',
    'Host': 'bkjw.whu.edu.cn',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh,en;q=0.9,zh-TW;q=0.8,zh-CN;q=0.7'
}

# if the output directory does not exist, create it
if not os.path.exists(CAPTCHA_IMAGE_FOLDER):
    os.makedirs(CAPTCHA_IMAGE_FOLDER)

i = 0
while i < 20:
    time.sleep( 1 )
    img = requests.get(img_url, headers=headers)
    f = open('./'+CAPTCHA_IMAGE_FOLDER+'/'+str(i)+'.jpg', 'ab')
    f.write(img.content)
    f.close()
    i += 1
    print(i)
print('end')



