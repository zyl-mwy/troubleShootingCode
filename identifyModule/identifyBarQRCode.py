# https://blog.csdn.net/qq_40962368/article/details/97498110
# https://blog.csdn.net/hxj0323/article/details/112969622

# 一、导入需要用到的包/库
import numpy as np
import imutils
import cv2

import datetime
import time
from pathlib import Path
import numpy as np
import cv2
from pyzbar import pyzbar

def get_qrcode_result(image_input, binary_max=230, binary_step=2):
    """
    获取二维码的结果
    :param image_input: 输入图片数据
    :param binary_max: 二值化的最大值
    :param binary_step: 每次递增的二值化步长
    :return: pyzbar 预测的结果
    """
    # 把输入图像灰度化
    if len(image_input.shape) >= 3:
        image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)

    # 获取自适配阈值
    binary, _ = cv2.threshold(image_input, 0, 255, cv2.THRESH_OTSU)

    # 二值化递增检测
    res = []
    while (binary < binary_max) and (len(res) == 0):
        binary, mat = cv2.threshold(image, binary, 255, cv2.THRESH_BINARY)
        res = pyzbar.decode(mat)
        binary += binary_step

    return res


# 二、转化为灰度图
image = cv2.imread(r'E:\my_project\troubleShootingProject\troubleShootingPic\identifyModule\2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(gray)
# cv2.imshow('gray',gray)
# cv2.waitKey()

# 三、使用Scharr算子进行边缘检测
ddepth = cv2.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX,gradY)
gradient = cv2.convertScaleAbs(gradient)

# 四、去除噪声
#         （1）模糊与阈值化处理
blurred = cv2.blur(gradient,(9, 9))
(_, thresh) = cv2.threshold(blurred, 231, 255, cv2.THRESH_BINARY)
# （2）形态学处理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# 五、确定检测轮廓，画出检测框
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
 
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)
 
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 识别二维码/条形码
image_dir = Path(r"E:\my_project\troubleShootingProject\troubleShootingPic\identifyModule")
total_image = 0
success_count = 0
for image_file in Path.iterdir(image_dir):
    if image_file.suffix not in [".jpg", ".png"]:
        # 非图片，跳过该文件
        continue

    # 使用 cv2.imdecode 可以读取中文路径下的图片
    image = cv2.imdecode(np.fromfile(Path(image_dir).joinpath(image_file), 
                                        dtype=np.uint8), 
                            cv2.IMREAD_COLOR)

    start_time = time.time()
    result = get_qrcode_result(image, binary_max=230, binary_step=2)

    print(f"Got {image_file} result: {result}, "
            f"using time : {datetime.timedelta(seconds=(time.time() - start_time))}")

    if len(result) > 0:
        success_count += 1
    total_image += 1

print(f"total image = {total_image}, success count = {success_count}")
