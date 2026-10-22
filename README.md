桥梁、建筑物缺陷检测数据集，裂缝、露筋、剥落、破损、泛碱5类缺陷检测 建筑桥梁缺陷检测 YOLOv11 桥梁、建筑物缺陷检测数据集，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81a2d8d1f0ff4331a0e87ab65186aa35.png)
3261张，yolo，voc，coco三种标注方式
图像尺寸:720*720
类别数量:5类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/959af9b570e844cb836dc90ee601d66d.png)

训练集图像数量:2936; 验证集图像数量:162； 测试集图像数量:163
类别名称: 每一类图像数 ，每一类标注数
Crack-Detection - 裂缝检测：1391, 2208
Exposed Rebar - 钢筋外露：483, 663
Spalling - 混凝土剥落：484, 675
Break - 断裂：472, 493
Efflorescence - 泛碱：422, 2143
image num: 3261![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8b34193c6f68447eb9e85daaf709c1c6.png)


2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9279fc37188945db97e6a290a1221866.png)

3.qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8105be8b589a43868eefca09bd215c58.png)



```python
import cv2
import numpy as np

def remove_watermark(image_path, output_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([12, 30, 30])
    upper_yellow = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, result)

remove_watermark("input.jpg", "output_no_wm.jpg")
```
>平台限制无法直接输出图片文件，本地运行，检测框、类别标签、PR曲线线条全部保留，仅清除水印文字。

# 建筑桥梁缺陷检测 YOLOv11![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d3c2d492aced4a8fa9bd807fac27515c.png)

## 1、配置文件 bridge_defect.yaml
```yaml
path: ./bridge_defect_dataset
train: images/train
val: images/val
names:
  0: Crack‑Detection
  1: Exposed‑Rebar
  2: Spalling
  3: Break
  4: Efflorescence
```

## 2、训练 train.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/56a8f92637da40aea3e9cdb61a49c860.png)

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    model.train(
        data="bridge_defect.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0
    )
```

## 3、推理 predict.py
```python
from ultralytics import YOLO
model = YOLO(r"./runs/detect/train/weights/best.pt")

def detect(file):
    res = model(file, conf=0.45)
    for r in res:
        r.save("detect_out.jpg")

if __name__ == "__main__":
    detect("test.jpg")
```

## 4、极简PyQt5界面 gui.py
```python
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QPushButton,QLabel,QFileDialog
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO

class DefectDetectUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("建筑桥梁缺陷检测系统")
        self.resize(820,620)
        self.model = YOLO("./runs/detect/train/weights/best.pt")

        self.btn_img = QPushButton("打开图片",self)
        self.btn_img.setGeometry(30,20,130,40)
        self.btn_img.clicked.connect(self.open_img)

        self.show_label = QLabel(self)
        self.show_label.setGeometry(30,80,760,520)

    def open_img(self):
        f,_ = QFileDialog.getOpenFileName()
        if f:
            out = self.model(f,conf=0.45)
            out[0].save("gui_result.jpg")
            self.show_label.setPixmap(QPixmap("gui_result.jpg").scaled(self.show_label.size()))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DefectDetectUI()
    win.show()
    sys.exit(app.exec_())
```

## 环境
```bash
pip install ultralytics pyqt5 opencv-python
```
功能检测类别：裂缝、露筋、剥落、破损、泛碱5类缺陷检测，图片检测GUI。


