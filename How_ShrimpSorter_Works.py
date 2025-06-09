import cv2
import time
import datetime
import threading
import queue
import RPi.GPIO as GPIO
import argparse
from ultralytics import YOLO

class ShrimpSortingSystem:
    def __init__(self, use_video_file=None):
        # System variables
        self.frame_width = 640
        self.frame_height = 480
        self.running = True
        self.use_video_file = use_video_file
        
        # ค่าพื้นที่สำหรับแยกขนาดกุ้ง
        self.size_thresholds = {
            "small": 32519.3,  # พื้นที่น้อยกว่า 32519.3 pixels² = Small
            "medium": 48045.8  # พื้นที่ระหว่าง 32519.3-48045.8 pixels² = Medium
            # พื้นที่มากกว่า 48045.8 pixels² = Large
        }
        
        # GPIO setup ใช้แบบเดิมไม่มีการเปลี่ยนแปลง
        GPIO.setmode(GPIO.BOARD)
        
        # ใช้ servo configs เดิมแต่เปลี่ยนชื่อวัตถุเป็นขนาดของกุ้ง
        self.servo_configs = {
            "small": {
                "pin": 11,
                "initial_angle": 14,
                "target_angle": 90,
                "hold_time": 2.0,
                "delay": 2.0
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
                "initial_angle": 11,
                "target_angle": 90,
                "hold_time": 2.0,
                "delay": 6.0
            }
        }
        
        # สร้าง PWM objects สำหรับแต่ละ servo (คงเดิม)
        self.servos = {}
        for shrimp_size, config in self.servo_configs.items():
            pin = config["pin"]
            GPIO.setup(pin, GPIO.OUT)
            servo = GPIO.PWM(pin, 50)  # 50Hz pulse
            servo.start(0)
            self.servos[shrimp_size] = servo
            
            # ตั้งค่า servo ไปที่องศาเริ่มต้น
            print(f"Setting {shrimp_size} shrimp servo to initial position: {config['initial_angle']} degrees")
            initial_duty = 2 + (config["initial_angle"] / 18)
            servo.ChangeDutyCycle(initial_duty)
            time.sleep(0.5)
            servo.ChangeDutyCycle(0)  # หยุด PWM เพื่อป้องกัน jitter

        # เปลี่ยนเป็นใช้โมเดลที่เทรนสำหรับกุ้ง
        self.model = YOLO("/home/project/Desktop/ShrimpDetection last.pt")  # เปลี่ยนเป็นโมเดลที่เทรนสำหรับกุ้ง
        self.confidence_threshold = 0.6
        
        # กำหนดกล้องหรือไฟล์วิดีโอตามตัวเลือก
        if self.use_video_file:
            self.cap = cv2.VideoCapture(self.use_video_file)
            print(f"Using video file: {self.use_video_file}")
        else:
            self.cap = cv2.VideoCapture(0)
            self.setup_camera()
            print("Using real-time camera")
        
        # Initialize object tracking variables
        self.shrimp_counts = {size: 0 for size in self.servo_configs.keys()}
        self.tracked_objects = {}  # เก็บข้อมูลวัตถุที่กำลังติดตาม
        
        # ปรับปรุงระบบการจัดการเฟรมและการประมวลผล
        self.display_frame = None  # เฟรมล่าสุดที่ใช้สำหรับแสดงผล
        self.current_detections = None  # ผลลัพธ์การตรวจจับล่าสุดที่ใช้สำหรับแสดงผล
        self.frame_lock = threading.Lock()  # ล็อคสำหรับการเข้าถึงเฟรมและผลลัพธ์การตรวจจับ
        
        # ลดระยะเวลาในการรอผลลัพธ์การตรวจจับ
        self.detection_interval = 0.05  # ลดลงจาก 0.1
        self.last_detection_time = 0

    def setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def determine_shrimp_size(self, box):
        """คำนวณขนาดของกุ้งจากพื้นที่ของกรอบ"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)  # คำนวณพื้นที่เป็น pixels²
        
        if area < self.size_thresholds["small"]:
            return "small"
        elif area < self.size_thresholds["medium"]:
            return "medium"
        else:
            return "large"

    def move_servo(self, shrimp_size):
        """ควบคุม servo ตามขนาดของกุ้ง - คงไว้ตามเดิม"""
        try:
            config = self.servo_configs[shrimp_size]
            initial_angle = config["initial_angle"]
            target_angle = config["target_angle"]
            hold_time = config["hold_time"]
            delay = config["delay"]
            servo = self.servos[shrimp_size]
            
            # ส่วนการเคลื่อนที่ servo ไปยังเป้าหมาย
            print(f"Moving {shrimp_size} shrimp servo to {target_angle} degrees")
            target_duty = 2 + (target_angle / 18)  # Convert angle to duty cycle
            servo.ChangeDutyCycle(target_duty)
            time.sleep(0.5)  # รอให้ servo เคลื่อนที่ไปถึงตำแหน่ง
            servo.ChangeDutyCycle(0)  # หยุด PWM เพื่อป้องกัน jitter
            
            # ค้างไว้ที่ตำแหน่งเป้าหมายตามเวลาที่กำหนด
            print(f"Holding {shrimp_size} shrimp servo at {target_angle} degrees for {hold_time} seconds")
            time.sleep(hold_time)
            
            # รอตามเวลา delay
            time.sleep(delay - hold_time if delay > hold_time else 0)
            
            # ส่ง servo กลับไปที่ตำแหน่งเริ่มต้น
            print(f"Returning {shrimp_size} shrimp servo to initial position: {initial_angle} degrees")
            initial_duty = 2 + (initial_angle / 18)  # Convert angle to duty cycle
            servo.ChangeDutyCycle(initial_duty)
            time.sleep(0.5)  # รอให้ servo เคลื่อนที่ไปถึงตำแหน่งเริ่มต้น
            servo.ChangeDutyCycle(0)  # หยุด PWM เพื่อป้องกัน jitter
            
        except Exception as e:
            print(f"Servo error for {shrimp_size} shrimp: {e}")

    def is_object_in_frame(self, box):
        """ตรวจสอบว่าวัตถุอยู่ในเฟรมหรือไม่"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return (0 <= x1 <= self.frame_width and 
                0 <= x2 <= self.frame_width and
                0 <= y1 <= self.frame_height and 
                0 <= y2 <= self.frame_height)

    def process_detections(self, frame, results):
        if results and len(results) > 0:
            current_time = time.time()
            active_tracks = set()  # เก็บ ID ที่เจอในเฟรมปัจจุบัน
            
            for result in results:
                for box in result.boxes:
                    if box.id is None:
                        continue
                        
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]  # ตอนนี้ class_name ควรจะเป็น "shrimp"
                    track_id = int(box.id[0])
                    conf = float(box.conf[0])
                    
                    # ตรวจสอบความเชื่อมั่น
                    if conf < self.confidence_threshold:
                        continue

                    # กำหนดขนาดของกุ้ง
                    shrimp_size = self.determine_shrimp_size(box)
                    unique_id = f"{class_name}_{track_id}"
                    
                    # ตรวจสอบว่าวัตถุอยู่ในเฟรมหรือไม่
                    if not self.is_object_in_frame(box):
                        if unique_id in self.tracked_objects:
                            del self.tracked_objects[unique_id]
                        continue
                    
                    active_tracks.add(unique_id)
                    
                    # จัดการวัตถุใหม่หรืออัพเดตวัตถุที่มีอยู่
                    if unique_id not in self.tracked_objects:
                        # วัตถุใหม่หรือวัตถุที่กลับเข้ามาในเฟรม
                        self.tracked_objects[unique_id] = {
                            'class': class_name,
                            'size': shrimp_size,
                            'last_seen': current_time,
                            'processed': False
                        }
                    else:
                        # อัพเดตขนาดและเวลาที่เห็นล่าสุด
                        self.tracked_objects[unique_id]['size'] = shrimp_size
                        self.tracked_objects[unique_id]['last_seen'] = current_time
                    
                    # ประมวลผลวัตถุที่ยังไม่ได้ประมวลผล
                    if not self.tracked_objects[unique_id]['processed']:
                        self.tracked_objects[unique_id]['processed'] = True
                        self.shrimp_counts[shrimp_size] += 1
                        print(f"Processing {shrimp_size} shrimp (ID: {track_id})")
                        
                        thread = threading.Thread(
                            target=self.move_servo,
                            args=(shrimp_size,)
                        )
                        thread.start()
            
            # ลบวัตถุที่ไม่ได้เจอในเฟรมปัจจุบันและไม่ได้เห็นมานาน
            for unique_id in list(self.tracked_objects.keys()):
                if (unique_id not in active_tracks and 
                    current_time - self.tracked_objects[unique_id]['last_seen'] > 1.0):
                    del self.tracked_objects[unique_id]

    def draw_boxes(self, frame, results):
        if results:
            for result in results:
                for box in result.boxes:
                    if box.id is None:
                        continue
                        
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0])
                    class_name = self.model.names[cls]
                    
                    if conf >= self.confidence_threshold:
                        # คำนวณขนาดของกุ้ง
                        shrimp_size = self.determine_shrimp_size(box)
                        unique_id = f"{class_name}_{track_id}"
                        processed = self.tracked_objects.get(unique_id, {}).get('processed', False)
                        
                        # คำนวณพื้นที่สำหรับแสดงในกรอบ
                        area = (x2 - x1) * (y2 - y1)
                        
                        # สีกรอบตามสถานะการประมวลผล
                        color = (0, 255, 0) if processed else (255, 165, 0)
                        
                        # วาดกรอบและจุดกึ่งกลาง
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                        
                        # แสดงข้อความที่ปรับปรุงแล้ว - เพิ่มขนาดและพื้นที่
                        label = f"{class_name} ({shrimp_size}) ID:{track_id} Area:{area:.1f}px²"
                        
                        # วาดพื้นหลังข้อความเพื่อให้อ่านง่ายขึ้น
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(
                            frame, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), 
                            color, 
                            -1
                        )
                        cv2.putText(
                            frame, 
                            label, 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            2
                        )

        # แสดงจำนวนการนับด้วยพื้นหลังสีเพื่อลดการกระพริบ
        y_pos = 30
        for size, count in self.shrimp_counts.items():
            text = f"{size} shrimp: {count}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, 
                (5, y_pos - text_height - 5), 
                (15 + text_width, y_pos + 5), 
                (255, 255, 255), 
                -1
            )
            cv2.putText(
                frame, 
                text, 
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 0, 0), 
                2
            )
            y_pos += 30

        # แสดงข้อมูลเกณฑ์ขนาดในรูปแบบที่กระชับมากขึ้น
        # ย้ายไปอยู่ที่มุมล่างซ้าย และลดขนาดตัวอักษร
        corner_x = 10
        corner_y = self.frame_height - 10  # เริ่มจากด้านล่างขึ้นมา
        
        # ทำพื้นหลังสำหรับข้อความให้เป็นแบบโปร่งใสเพื่อให้อ่านง่ายแต่ไม่บัง
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (corner_x - 5, corner_y - 75), 
            (corner_x + 300, corner_y + 5), 
            (0, 0, 0), 
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # 60% โปร่งใส
        
        # ลดขนาดข้อความ
        font_size = 0.4
        text_thickness = 1
        line_height = 18
        
        # เพิ่มข้อมูลเกณฑ์ขนาดในบรรทัดเดียว
        cv2.putText(
            frame, 
            f"Size: S< {int(self.size_thresholds['small'])}, M: {int(self.size_thresholds['small'])}-{int(self.size_thresholds['medium'])}, L> {int(self.size_thresholds['medium'])} px²", 
            (corner_x, corner_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_size, 
            (255, 255, 255), 
            text_thickness
        )

    def run(self):
        print("Starting Shrimp Sorting System...")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Servo initial positions:")
        for size, config in self.servo_configs.items():
            print(f"  - {size} shrimp: {config['initial_angle']} degrees")
        print("Size thresholds (pixels²):")
        print(f"  - Small: < {self.size_thresholds['small']}")
        print(f"  - Medium: {self.size_thresholds['small']} - {self.size_thresholds['medium']}")
        print(f"  - Large: > {self.size_thresholds['medium']}")
        time.sleep(2)
        
        try:
            while self.running:
                start_time = time.time()  # เริ่มจับเวลาการประมวลผลแต่ละเฟรม
                
                # อ่านเฟรมจากกล้องหรือวิดีโอ
                ret, frame = self.cap.read()
                if not ret:
                    # ถ้าเป็นไฟล์วิดีโอและเล่นจบแล้ว ให้เริ่มเล่นใหม่
                    if self.use_video_file:
                        print("Video ended, restarting...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Failed to grab frame")
                        break
                
                # ปรับขนาดเฟรม
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # ตรวจจับวัตถุในเฟรมปัจจุบันโดยตรง (ไม่ผ่านคิว)
                # ให้โมเดลประมวลผลเฟรมปัจจุบันโดยตรงเพื่อลดความล่าช้า
                current_time = time.time()
                if current_time - self.last_detection_time >= self.detection_interval:
                    # เรียกใช้โมเดล YOLO โดยตรงกับเฟรมปัจจุบัน (ไม่ผ่านคิว)
                    results = self.model.track(frame, persist=True, conf=self.confidence_threshold)
                    
                    # ประมวลผลการตรวจจับวัตถุ
                    with self.frame_lock:
                        self.process_detections(frame, results)
                        self.current_detections = results
                    
                    self.last_detection_time = current_time

                # สร้างเฟรมสำหรับแสดงผล
                display_frame = frame.copy()
                
                # วาด bounding boxes สำหรับเฟรมปัจจุบันโดยใช้ผลลัพธ์ล่าสุด
                with self.frame_lock:
                    if self.current_detections is not None:
                        self.draw_boxes(display_frame, self.current_detections)
                    else:
                        self.draw_boxes(display_frame, None)
                
                # แสดงค่า FPS และแหล่งที่มาของวิดีโอ
                elapsed_time = time.time() - start_time
                fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(
                    display_frame, 
                    f"FPS: {fps:.1f}", 
                    (self.frame_width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 255), 
                    2
                )
                
                # แสดงว่ากำลังใช้โหมดไหน (วิดีโอ/กล้อง)
                source_type = "Video File" if self.use_video_file else "Camera"
                cv2.putText(
                    display_frame, 
                    f"Source: {source_type}", 
                    (self.frame_width - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 255), 
                    2
                )
                
                # ชะลอการเล่นวิดีโอ - เพิ่มการหน่วงเวลาถ้าเป็นไฟล์วิดีโอ
                if self.use_video_file:
                    # ลดเวลาหน่วง เพื่อให้การตรวจจับแม่นยำขึ้น
                    time.sleep(0.03)  # 30 มิลลิวินาที แทน 100 มิลลิวินาที
                
                # แสดงผลเฟรม
                cv2.imshow("Shrimp Sorting System", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nSummary ({timestamp})")
        for size, count in self.shrimp_counts.items():
            print(f"{size} shrimp: {count}")
        
        # หยุด PWM และทำความสะอาด GPIO
        for servo in self.servos.values():
            servo.stop()
        GPIO.cleanup()
            
        self.cap.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Shrimp Sorting System')
    parser.add_argument('--video', type=str, help='Path to video file. If not provided, camera will be used.')
    return parser.parse_args()

if __name__ == "__main__":
    # ระบุพาธของวิดีโอโดยตรงที่นี่ (ถ้าต้องการใช้วิดีโอไฟล์)
    video_path = "/home/project/Desktop/Test.mp4"  # เปลี่ยนเป็นพาธของวิดีโอที่คุณต้องการใช้
    
    # ถ้าต้องการใช้กล้องแบบเรียลไทม์ ให้กำหนดเป็น None
    #video_path = None
    
    # เรียกใช้คลาส ShrimpSortingSystem โดยส่งพาธของวิดีโอเข้าไปโดยตรง
    sorter = ShrimpSortingSystem(use_video_file=video_path)
    sorter.run()