import cv2
import numpy as np
import datetime

# 读取原始图像
img = cv2.imread('test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 形态学去噪
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建图像副本
output = img.copy()
min_area = 1000
object_index = 0

# 打开一个txt文件用于写入
with open('output.txt', 'w') as file:
    file.write(f"Detection Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 轮廓编号
        object_index += 1

        # 中心点
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        # 最小外接矩形获取方向角度
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        if angle < -45:
            angle += 90  # 修正角度到[-45, 45]范围

        # 获取边界坐标
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 写入txt文件
        file.write(f"Target {object_index}: Center ({cx}, {cy}), Angle {angle:.2f}°, "
                   f"Border coordinates {box.tolist()}\n")

        # 绘制中心点
        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)  # 红色圆点

        # 绘制箭头表示方向
        arrow_length = 50
        arrow_end_x = int(cx + arrow_length * np.cos(np.deg2rad(angle)))
        arrow_end_y = int(cy + arrow_length * np.sin(np.deg2rad(angle)))
        cv2.arrowedLine(output, (cx, cy), (arrow_end_x, arrow_end_y), (255, 0, 0), 2, tipLength=0.3)  # 蓝色箭头

        # 绘制边界框
        cv2.drawContours(output, [box], 0, (0, 255, 0), 2)  # 绿色边界框

# 在左上角显示总数量
cv2.putText(output, f"Total Count: {object_index}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # 黄色文本

# 保存标注图像
cv2.imwrite('result_with_total_count.png', output)
