# https://blog.51cto.com/u_16099329/7741573
# https://blog.csdn.net/qq_38017966/article/details/118724459
import cv2
import numpy as np
import easyocr

#设置识别中英文两种语言
# reader = easyocr.Reader(['ch_sim','en'], gpu = False) # need to run only once to load model into memory
# result = reader.readtext(r"d:\Desktop\4A34A16F-6B12-4ffc-88C6-FC86E4DF6912.png", detail = 0)
# print(result)


# 1.读取文件，并专成灰度图
imagePath = r'E:\my_project\troubleShootingProject\troubleShootingPic\identifyModule\1.png'
img = cv2.imread(imagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey()

# 2.形态学变换的预处理，得到可以查找矩形的图片
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
# cv2.imshow('sobel',sobel)
# cv2.waitKey()

# 3.二值化并设置膨胀和腐蚀操作的核函数
#二值化
ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
# 设置膨胀和腐蚀操作的核函数
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

# 4.膨胀一次，让轮廓突出
dilation = cv2.dilate(binary, element2, iterations = 1)
# cv2.imshow('dilation', dilation)
# cv2.waitKey()

# 5.腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
erosion = cv2.erode(dilation, element1, iterations = 1)
# cv2.imshow('erosion', erosion)
# cv2.waitKey()

# 6.再次膨胀，让轮廓明显一些
dilation2 = cv2.dilate(erosion, element2, iterations = 3)
# cv2.imshow('dilation2', dilation2)
# cv2.waitKey()

# 7.查找和筛选文字区域
region = []
#  查找轮廓
contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 利用以上函数可以得到多个轮廓区域，存在一个列表中。
#  筛选那些面积小的

for i in range(len(contours)):
    # 遍历所有轮廓
    # cnt是一个点集

    cnt = contours[i]

    # 计算该轮廓的面积
    area = cv2.contourArea(cnt)

    # 面积小的都筛选掉、这个1000可以按照效果自行设置
    if(area < 1000):
        continue

#     # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定
#     # 轮廓近似，作用很小
#     # 计算轮廓长度
#     epsilon = 0.001 * cv2.arcLength(cnt, True)

#     #
# #     approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 找到最小的矩形，该矩形可能有方向
    rect = cv2.minAreaRect(cnt)
    # 打印出各个矩形四个点的位置
    # print ("rect is: ")
    # print (rect)

    # box是四个点的坐标
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算高和宽
    height = abs(box[0][1] - box[2][1])
    width = abs(box[0][0] - box[2][0])

    # 筛选那些太细的矩形，留下扁的
    if(height > width * 1.3):
        continue

    region.append(box)
    # 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        print(box)
    # plt.imshow(img, 'brg')
    # plt.show()

# 弹窗显示
# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.imshow("img", img)

    # 带轮廓的图片
# cv2.waitKey(0)
# cv2.destroyAllWindows()
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        crop_img: object = img[y1:y1 + hight, x1:x1 + width]
        # print(r'E:\my_project\troubleShootingProject\troubleShootingCode\identifyModule\results/%d'%i+'.jpg')
        # im=cv2.imshow('crop_img', crop_img)
        # cv2.imwrite(r'E:\my_project\troubleShootingProject\troubleShootingCode\identifyModule\results/%d'%i+'.jpg',crop_img,[int(cv2.IMWRITE_JPEG_QUALITY),70])

#设置识别中英文两种语言
reader = easyocr.Reader(['ch_sim','en'], gpu = False) # need to run only once to load model into memory
result = reader.readtext(r"E:\my_project\troubleShootingProject\troubleShootingPic\identifyModule\results\0.jpg", detail = 0)
print(result)
