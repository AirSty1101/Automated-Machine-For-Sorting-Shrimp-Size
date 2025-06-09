# Shrimp Sorting System

## คำอธิบาย

ระบบคัดแยกขนาดกุ้งขาวอัตโนมัติที่ใช้ AI และ Computer Vision ในการจำแนกประเภทและคัดแยกกุ้งเป็น 3 ขนาด:
- **S (Small)**: กุ้งขนาดเล็ก
- **M (Medium)**: กุ้งขนาดกลาง  
- **L (Large)**: กุ้งขนาดใหญ่

ระบบจะทำการตรวจจับกุ้งด้วยโมเดล YOLO แล้วคำนวณขนาดจากพื้นที่ของ bounding box เพื่อจำแนกประเภท จากนั้นใช้ servo motor ควบคุมการคัดแยกตามขนาดที่ได้

## การเตรียมการและการติดตั้ง

### อุปกรณ์ที่ต้องใช้
- Raspberry Pi (รองรับ GPIO)
- กล้อง USB หรือ Raspberry Pi Camera
- Servo motors จำนวน 3 ตัว (สำหรับแต่ละขนาด S, M, L)
- โมเดล YOLO ที่เทรนสำหรับการตรวจจับกุ้ง

### Libraries ที่ต้องติดตั้ง
```bash
pip install opencv-python
pip install ultralytics
pip install RPi.GPIO
pip install numpy
```

### Libraries เพิ่มเติมสำหรับ Python
```python
import cv2
import time
import datetime
import threading
import queue
import argparse
import csv
import os
```

## การกำหนดค่าโมเดล

สามารถเปลี่ยนโมเดล YOLO ที่ใช้ได้ที่บรรทัดที่ 72:

```python
self.model = YOLO("/home/project/Desktop/ShrimpDetection last.pt")
```

### ตัวอย่างการเปลี่ยนโมเดล:
```python
# ใช้โมเดลที่เทรนเอง
self.model = YOLO("/path/to/your/custom_shrimp_model.pt")

# ใช้โมเดลจาก Ultralytics Hub
self.model = YOLO("your_model_id")

# ใช้โมเดลพื้นฐานของ YOLOv8
self.model = YOLO("yolov8n.pt")
```

## การเลือกแหล่งภาพ

### ใช้กล้อง (Real-time)
ที่บรรทัดที่ 688 ให้กำหนด:
```python
video_path = None
```

### ใช้ไฟล์วิดีโอ
ที่บรรทัดที่ 688 ให้กำหนดเส้นทางไฟล์:
```python
# ตัวอย่างการกำหนดเส้นทางวิดีโอ
video_path = "/home/project/Desktop/Test.mp4"
video_path = "/home/user/Videos/shrimp_test.avi"
video_path = "C:\\Users\\username\\Desktop\\test_video.mp4"  # สำหรับ Windows
```

## การปรับแก้ Size Threshold

สามารถปรับเกณฑ์การแยกขนาดได้ที่บรรทัดที่ 21-25:

```python
self.size_thresholds = {
    "small": 32519.3,  # พื้นที่น้อยกว่า 32519.3 pixels² = Small
    "medium": 48045.8  # พื้นที่ระหว่าง 32519.3-48045.8 pixels² = Medium
    # พื้นที่มากกว่า 48045.8 pixels² = Large
}
```

### ตัวอย่างการปรับแก้:
```python
# เกณฑ์ใหม่สำหรับกุ้งขนาดต่างกัน
self.size_thresholds = {
    "small": 25000.0,   # กุ้งเล็กกว่า 25,000 pixels²
    "medium": 40000.0   # กุ้งขนาดกลาง 25,000-40,000 pixels²
    # กุ้งใหญ่มากกว่า 40,000 pixels²
}
```

**หมายเหตุ**: ค่า threshold ได้มาจากการรันโค้ด `CheckPixel.py` เพื่อวิเคราะห์ขนาดจริงของกุ้งจากชุดข้อมูลที่เก็บไว้

## การกำหนดค่า Servo Motors

สามารถปรับแก้การตั้งค่า servo ได้ที่บรรทัดที่ 31-53:

```python
self.servo_configs = {
    "small": {
        "pin": 11,              # GPIO pin number
        "initial_angle": 13,    # มุมเริ่มต้น (องศา)
        "target_angle": 90,     # มุมเป้าหมายเมื่อทำงาน (องศา)
        "hold_time": 2.0,       # เวลาค้างที่มุมเป้าหมาย (วินาที)
        "delay": 2.0            # เวลารอรวม (วินาที)
    },
    "medium": {
        "pin": 13,
        "initial_angle": 8,
        "target_angle": 90,
        "hold_time": 2.0,
        "delay": 4.0
    },
    "large": {
        "pin": 15,
        "initial_angle": 10,
        "target_angle": 90,
        "hold_time": 2.0,
        "delay": 6.0
    }
}
```

### ตัวอย่างการปรับแก้:
```python
# ปรับแก้ servo สำหรับกุ้งขนาดเล็ก
"small": {
    "pin": 11,              # ใช้ GPIO pin 11
    "initial_angle": 0,     # เริ่มต้นที่ 0 องศา
    "target_angle": 45,     # หมุนไป 45 องศา
    "hold_time": 1.5,       # ค้างไว้ 1.5 วินาที
    "delay": 3.0            # รอ 3 วินาทีก่อนกลับ
}
```

## การปรับแก้ Confidence Threshold

สามารถปรับระดับความเชื่อมั่นในการตรวจจับได้ที่บรรทัดที่ 73:

```python
self.confidence_threshold = 0.6  # ความเชื่อมั่น 60%
```

### ตัวอย่างการปรับแก้:
```python
self.confidence_threshold = 0.5   # ลดเป็น 50% (ตรวจจับได้ง่ายขึ้น)
self.confidence_threshold = 0.8   # เพิ่มเป็น 80% (ตรวจจับแม่นยำขึ้น)
```

## การใช้งาน

### การเริ่มต้นโปรแกรม
```bash
python shrimp_sorting_system.py
```

### การทดสอบและดูผลการทำงาน
หากต้องการทดสอบระบบและดู Bounding Box การทำงานของการตรวจจับ โดยไม่บันทึกเป็น CSV file:

```bash
python How_ShrimpSorter_Works.py
```

**หมายเหตุ**: ไฟล์ `How_ShrimpSorter_Works.py` มีการทำงานเหมือนระบบหลักทุกประการ ได้แก่:
- การตรวจจับและจำแนกขนาดกุ้ง
- การแสดง Bounding Box และข้อมูลต่างๆ บนหน้าจอ
- การควบคุม Servo Motors
- การนับจำนวนกุ้งแต่ละขนาด

เพียงแต่จะไม่มีการบันทึกข้อมูลลงในไฟล์ CSV ทำให้เหมาะสำหรับการทดสอบและดูผลการทำงาน

### การหยุดการทำงาน
กดปุ่ม **'q'** เพื่อหยุดการทำงานและบันทึกข้อมูล

## ไฟล์ผลลัพธ์

เมื่อหยุดการทำงาน ระบบจะบันทึกข้อมูลเป็น CSV ไฟล์ 2 ประเภท:

### 1. ไฟล์ข้อมูลรายละเอียด
**ชื่อไฟล์**: `shrimp_sorting_data_YYYYMMDD_HHMMSS.csv`

**เนื้อหา**:
- timestamp: เวลาที่ตรวจจับ
- detection_time: เวลาในรูปแบบ Unix timestamp
- class_name: ชื่อคลาส (shrimp)
- shrimp_size: ขนาดกุ้ง (small/medium/large)
- track_id: หมายเลขติดตามวัตถุ
- confidence: ระดับความเชื่อมั่น
- area_pixels: พื้นที่เป็น pixels²
- box_x1, box_y1, box_x2, box_y2: พิกัด bounding box
- center_x, center_y: จุดกึ่งกลางวัตถุ
- processed_status: สถานะการประมวลผล (True/False)

### 2. ไฟล์สรุปผลลัพธ์
**ชื่อไฟล์**: `shrimp_sorting_summary_YYYYMMDD_HHMMSS.csv`

**เนื้อหา**:
- shrimp_size: ขนาดกุ้ง
- count: จำนวนที่นับได้
- timestamp: เวลาที่บันทึก

### ตัวอย่างผลลัพธ์:
```
Small shrimp: 15
Medium shrimp: 23
Large shrimp: 12
```

## การแก้ไขปัญหา

### ปัญหาที่พบบ่อย
1. **กล้องไม่ทำงาน**: ตรวจสอบการเชื่อมต่อและ permission
2. **Servo ไม่เคลื่อนที่**: ตรวจสอบการเชื่อมต่อ GPIO และ power supply
3. **ตรวจจับไม่แม่นยำ**: ปรับ confidence_threshold หรือเปลี่ยนโมเดล
4. **Size threshold ไม่เหมาะสม**: รัน CheckPixel.py เพื่อหาค่าที่เหมาะสม

### การตรวจสอบประสิทธิภาพ
- **FPS**: แสดงบนหน้าจอเพื่อตรวจสอบประสิทธิภาพ
- **Display FPS**: ความเร็วในการแสดงผล
- **Processing FPS**: ความเร็วในการประมวลผล

## หมายเหตุ

- ระบบรองรับการทำงานแบบ real-time และการเล่นไฟล์วิดีโอ
- ใช้ multithreading เพื่อเพิ่มประสิทธิภาพ
- มีการติดตามวัตถุ (object tracking) เพื่อป้องกันการนับซ้ำ
- บันทึกข้อมูลแบบ real-time เพื่อความปลอดภัยของข้อมูล