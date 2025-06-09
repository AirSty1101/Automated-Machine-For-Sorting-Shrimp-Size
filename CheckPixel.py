import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import glob

class ShrimpSizeCalibrator:
    def __init__(self, model_path="yolov12.pt", confidence=0.7):
        # กำหนดค่าเริ่มต้น
        self.confidence_threshold = confidence
        
        # โหลดโมเดล YOLO
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        # ตัวแปรเก็บข้อมูลขนาด
        self.size_data = {'small': [], 'medium': [], 'large': []}
    
    def get_size_color(self, size):
        """สีสำหรับแต่ละขนาด"""
        if size == "small":
            return (0, 255, 0)  # สีเขียว
        elif size == "medium":
            return (0, 165, 255)  # สีส้ม
        else:  # large
            return (0, 0, 255)  # สีแดง
    
    def calculate_stats(self, data):
        """คำนวณสถิติพื้นฐาน"""
        if not data:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "count": len(data),
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        }
    
    def process_image(self, image_path, size_category):
        """ประมวลผลภาพและบันทึกข้อมูลขนาด"""
        print(f"กำลังประมวลผล: {image_path}")
        
        # อ่านภาพ
        image = cv2.imread(image_path)
        if image is None:
            print(f"ไม่สามารถอ่านไฟล์ภาพ: {image_path}")
            return None
        
        # ปรับขนาดภาพ (ถ้าจำเป็น)
        if image.shape[0] != 640 or image.shape[1] != 640:
            image = cv2.resize(image, (640, 640))
        
        # ตรวจจับกุ้งด้วย YOLO
        results = self.model(image, conf=self.confidence_threshold)
        
        # เตรียมภาพสำหรับแสดงผล
        display_image = image.copy()
        
        # เก็บข้อมูลพื้นที่ของกุ้งทั้งหมดที่ตรวจพบ
        detected_areas = []
        
        # วิเคราะห์ผลการตรวจจับ
        if results and len(results) > 0:
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    conf = float(box.conf[0])
                    
                    # เน้นการตรวจจับกุ้งหรือวัตถุที่สนใจ
                    detect_class = "all"  # หรือเปลี่ยนเป็น "shrimp" ถ้าโมเดลสามารถตรวจจับกุ้งโดยเฉพาะ
                    
                    if (detect_class == "all" or class_name == detect_class) and conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # เก็บข้อมูลพื้นที่
                        detected_areas.append(area)
                        
                        # แสดงกรอบและขนาดพื้นที่
                        color = self.get_size_color(size_category)
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                        
                        # แสดงข้อมูลขนาด
                        label = f"{class_name} - Area: {area} px² (W:{width} x H:{height})"
                        cv2.putText(
                            display_image, 
                            label, 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            2
                        )
        
        # กรณีตรวจพบมากกว่า 1 ตัวในภาพ เลือกเฉพาะตัวที่ใหญ่ที่สุด
        # (สมมติว่าเราต้องการวัดขนาดกุ้งตัวเดียวต่อภาพ)
        if detected_areas:
            largest_area = max(detected_areas)
            self.size_data[size_category].append(largest_area)
            return display_image, largest_area
        else:
            print(f"ไม่พบกุ้งในภาพ: {image_path}")
            return display_image, None
    
    def batch_process_images(self, folders):
        """ประมวลผลภาพทั้งหมดในโฟลเดอร์ที่กำหนด"""
        print("\n===== กำลังประมวลผลภาพเพื่อสอบเทียบขนาด =====")
        
        # ประมวลผลภาพแต่ละขนาด
        for size_category, folder_path in folders.items():
            print(f"\nกำลังประมวลผลภาพขนาด {size_category.upper()} จาก {folder_path}")
            
            # หาไฟล์ภาพในโฟลเดอร์
            image_files = []
            for ext in ['jpg', 'jpeg', 'png']:
                image_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))
                image_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext.upper()}")))
            
            if not image_files:
                print(f"ไม่พบไฟล์ภาพใน {folder_path}")
                continue
            
            print(f"พบภาพทั้งหมด {len(image_files)} ไฟล์")
            
            # สร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์ (ถ้ายังไม่มี)
            output_folder = os.path.join(os.path.dirname(folder_path), f"results_{size_category}")
            os.makedirs(output_folder, exist_ok=True)
            
            # ประมวลผลแต่ละภาพ
            for i, image_path in enumerate(image_files):
                result_image, area = self.process_image(image_path, size_category)
                
                if result_image is not None:
                    # บันทึกภาพผลลัพธ์
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(output_folder, f"result_{filename}")
                    cv2.imwrite(output_path, result_image)
                    
                    # แสดงความคืบหน้า
                    if area is not None:
                        print(f"ประมวลผล {i+1}/{len(image_files)}: {filename} - พื้นที่ = {area:.1f} pixels²")
                    else:
                        print(f"ประมวลผล {i+1}/{len(image_files)}: {filename} - ไม่พบกุ้ง")
        
        # เมื่อประมวลผลเสร็จสิ้น แสดงผลสรุป
        self.show_summary()
        
        # สร้างกราฟแสดงผลการกระจายตัว
        self.plot_distribution()
        
    def plot_distribution(self):
        """สร้างกราฟแสดงการกระจายตัวของขนาดกุ้งแต่ละประเภท"""
        plt.figure(figsize=(12, 8))
        
        colors = {'small': 'green', 'medium': 'orange', 'large': 'red'}
        
        for size, data in self.size_data.items():
            if data:
                plt.hist(data, bins=20, alpha=0.7, label=f'{size.upper()} (n={len(data)})', color=colors[size])
        
        # คำนวณและแสดงเส้นแบ่งขนาด (threshold)
        if all(len(self.size_data[size]) > 0 for size in ['small', 'medium', 'large']):
            small_mean = np.mean(self.size_data['small'])
            medium_mean = np.mean(self.size_data['medium'])
            large_mean = np.mean(self.size_data['large'])
            
            small_medium_threshold = (small_mean + medium_mean) / 2
            medium_large_threshold = (medium_mean + large_mean) / 2
            
            plt.axvline(x=small_medium_threshold, color='blue', linestyle='--', 
                       label=f'Small-Medium Threshold: {small_medium_threshold:.1f}')
            plt.axvline(x=medium_large_threshold, color='purple', linestyle='--', 
                       label=f'Medium-Large Threshold: {medium_large_threshold:.1f}')
        
        plt.title('Distribution of area size of each type of shrimp')
        plt.xlabel('Area (pixels²)')
        plt.ylabel('Amount')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # บันทึกกราฟเป็นไฟล์
        plt.savefig('shrimp_size_distribution.png')
        print("\nบันทึกกราฟการกระจายตัวเป็น shrimp_size_distribution.png")
        plt.close()
    
    def show_summary(self):
        """แสดงผลสรุปและคำแนะนำ"""
        print("\n===== สรุปผลการสอบเทียบ =====")
        all_data_valid = True
        
        for size in ['small', 'medium', 'large']:
            data = self.size_data[size]
            if not data:
                print(f"ไม่พบข้อมูลสำหรับขนาด {size.upper()}")
                all_data_valid = False
                continue
            
            stats = self.calculate_stats(data)
            print(f"{size.upper()}:")
            print(f"  จำนวนตัวอย่าง: {stats['count']}")
            print(f"  ค่าเฉลี่ย: {stats['mean']:.2f} pixels²")
            print(f"  ส่วนเบี่ยงเบนมาตรฐาน: {stats['std']:.2f}")
            print(f"  ค่าต่ำสุด: {stats['min']:.2f} pixels²")
            print(f"  ค่าสูงสุด: {stats['max']:.2f} pixels²")
        
        if all_data_valid:
            # คำนวณจุดตัด (threshold) ระหว่างขนาด
            small_mean = np.mean(self.size_data['small'])
            medium_mean = np.mean(self.size_data['medium'])
            large_mean = np.mean(self.size_data['large'])
            
            small_medium_threshold = (small_mean + medium_mean) / 2
            medium_large_threshold = (medium_mean + large_mean) / 2
            
            print("\n===== ค่าแนะนำสำหรับการตั้งค่าในโค้ดหลัก =====")
            print("เพิ่มค่าต่อไปนี้ในคลาส ShrimpSortingSystem ในฟังก์ชัน __init__:")
            print(f"""
self.size_thresholds = {{
    "small": {small_medium_threshold:.1f},  # พื้นที่น้อยกว่า {small_medium_threshold:.1f} pixels² = ขนาดเล็ก
    "medium": {medium_large_threshold:.1f}  # พื้นที่ระหว่าง {small_medium_threshold:.1f}-{medium_large_threshold:.1f} pixels² = ขนาดกลาง
                                  # พื้นที่มากกว่า {medium_large_threshold:.1f} pixels² = ขนาดใหญ่
}}
""")
            
            # ตรวจสอบความแม่นยำของการคัดแยก
            print("\n===== ตรวจสอบความแม่นยำของเกณฑ์ที่แนะนำ =====")
            small_correct = sum(1 for area in self.size_data['small'] if area < small_medium_threshold)
            small_accuracy = (small_correct / len(self.size_data['small'])) * 100 if self.size_data['small'] else 0
            
            medium_correct = sum(1 for area in self.size_data['medium'] 
                               if small_medium_threshold <= area < medium_large_threshold)
            medium_accuracy = (medium_correct / len(self.size_data['medium'])) * 100 if self.size_data['medium'] else 0
            
            large_correct = sum(1 for area in self.size_data['large'] if area >= medium_large_threshold)
            large_accuracy = (large_correct / len(self.size_data['large'])) * 100 if self.size_data['large'] else 0
            
            print(f"ความแม่นยำในการคัดแยกขนาดเล็ก: {small_accuracy:.1f}% ({small_correct}/{len(self.size_data['small'])})")
            print(f"ความแม่นยำในการคัดแยกขนาดกลาง: {medium_accuracy:.1f}% ({medium_correct}/{len(self.size_data['medium'])})")
            print(f"ความแม่นยำในการคัดแยกขนาดใหญ่: {large_accuracy:.1f}% ({large_correct}/{len(self.size_data['large'])})")
            
            average_accuracy = (small_accuracy + medium_accuracy + large_accuracy) / 3
            print(f"ความแม่นยำเฉลี่ย: {average_accuracy:.1f}%")
            
            if average_accuracy < 90:
                print("\nข้อควรระวัง: ความแม่นยำต่ำกว่า 90% อาจเกิดจาก:")
                print("  1. มีความคาบเกี่ยวของขนาดกุ้งมากเกินไป")
                print("  2. ตัวอย่างมีความแปรปรวนสูง")
                print("  3. จำนวนตัวอย่างไม่เพียงพอ")
                print("ควรพิจารณาเก็บตัวอย่างเพิ่มหรือปรับปรุงวิธีการวัด")
        else:
            print("\nไม่สามารถคำนวณค่าแนะนำได้เนื่องจากข้อมูลไม่ครบ")
            print("กรุณาเก็บข้อมูลให้ครบทั้ง 3 ขนาด (small, medium, large)")

# ในส่วนของ if __name__ == "__main__":
if __name__ == "__main__":
    # กำหนดพาธเริ่มต้น (ปรับให้เป็นพาธจริงๆ ของคุณ)
    default_small_folder = "Image Scerw\Small"
    default_medium_folder = "Image Scerw\Medium"
    default_large_folder = "Image Scerw\Large"
    
    # ตรวจสอบว่ามีอาร์กิวเมนต์ส่งมาหรือไม่
    try:
        # รับพารามิเตอร์จากคำสั่ง (command line)
        parser = argparse.ArgumentParser(description='เครื่องมือสอบเทียบขนาดกุ้งจากภาพที่มีอยู่')
        parser.add_argument('--model', type=str, default='Nut\last.pt', 
                            help='ที่อยู่ของโมเดล YOLO (default: yolov8s.pt)')
        parser.add_argument('--conf', type=float, default=0.75,
                            help='ค่าความเชื่อมั่นขั้นต่ำ (default: 0.6)')
        parser.add_argument('--small', type=str, default=default_small_folder,
                            help='โฟลเดอร์ที่เก็บภาพกุ้งขนาดเล็ก')
        parser.add_argument('--medium', type=str, default=default_medium_folder,
                            help='โฟลเดอร์ที่เก็บภาพกุ้งขนาดกลาง')
        parser.add_argument('--large', type=str, default=default_large_folder,
                            help='โฟลเดอร์ที่เก็บภาพกุ้งขนาดใหญ่')
        
        args = parser.parse_args()
        
        # เตรียมข้อมูลโฟลเดอร์
        folders = {
            'small': args.small,
            'medium': args.medium,
            'large': args.large
        }
        
        # สร้างและเริ่มตัวสอบเทียบ
        calibrator = ShrimpSizeCalibrator(model_path=args.model, confidence=args.conf)
        calibrator.batch_process_images(folders)
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {e}")