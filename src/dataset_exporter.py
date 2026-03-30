import datetime
import os
import time
import json
import rclpy

from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image 
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

from dataset_manager import DatasetManager
from coco_converter import CocoConverter
import utils


class DatasetExporter(Node): 
  """將圖像傳入進行影像訂閱"""

  def __init__(self):
    super().__init__('dataset_exporter')
    self.start_time = time.time()
    now = datetime.datetime.now()
    self.time_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    self.last_save_time = 0
    self.save_interval = 0.5

    self.bridge = CvBridge()
    self.converter = CocoConverter()
    self.dataset = DatasetManager(time_stamp=self.time_str)
    self.semantic_id_to_name = {}

    self.rgb_sub = Subscriber(self, Image, "/rgb")
    self.inst_sub = Subscriber(self, Image, "/instance_segmentation")
    self.sem_sub = Subscriber(self, Image, "/semantic_segmentation")

    self.semantic_label_sub = self.create_subscription(
         String,
         "/semantic_labels",
         self.semantic_callback,
         10
      )

    self.ts = ApproximateTimeSynchronizer(
      [self.rgb_sub, self.inst_sub, self.sem_sub],
       queue_size = 10,
       slop = 0.1
    )

    self.ts.registerCallback(self.synced_callback)
  
  def synced_callback(self, rgb_msg, inst_msg, sem_msg):
     current_time = time.time()

     if current_time - self.start_time > 30:
        print("⏰ Reached 30 seconds, shutting down...")
        self.destroy_node()
        rclpy.shutdown()
        return
     
     if current_time - self.last_save_time < self.save_interval:
         return

     self.last_save_time = current_time
     
     if len(self.semantic_id_to_name) < 3:
         print("⚠️ semantic map 還沒準備好")
         return
     
     timestamp = f"{rgb_msg.header.stamp.sec}_{rgb_msg.header.stamp.nanosec}"

     rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "passthrough")
     inst = self.bridge.imgmsg_to_cv2(inst_msg, "passthrough")
     sem = self.bridge.imgmsg_to_cv2(sem_msg, "passthrough")

     print(f"inst unique: {np.unique(inst)}")
     print(f"sem  unique: {np.unique(sem)}")

     self.process(rgb, inst, sem, timestamp)
     self.process_inst_seg(inst, sem, timestamp)

  # 處理 rgb, bbox2d
  def process(self, rgb, inst, sem, timestamp):
    try:
        instance = self.converter.extract_instances(inst)
        inst_to_class = self.build_inst_to_class(inst, sem)
        merged_instances = self.merge_instances_by_class(instance, inst_to_class, sem)

        # 如果是 RGBA → 轉 BGR
        if rgb.shape[-1] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
        # 存rgb
        rgb_file_name = f"rgb_frame_{timestamp}.png"

        rgb_dir_path = f"data_storage/rgb_image/{self.dataset.time_str}"
        os.makedirs(rgb_dir_path, exist_ok=True)

        rgb_image_path = f"{rgb_dir_path}/{rgb_file_name}"

        cv2.imwrite(rgb_image_path, rgb)

        # 畫 bbox
        bbox_img = rgb.copy()

        for class_name, mask in merged_instances:

           color = utils.get_color(hash(class_name) % 1000)
           color = tuple(int(c) for c in color)
           x, y, w, h = self.converter.mask_to_bbox(mask)
           h_img, w_img = mask.shape

           if w * h >= 0.95 * (w_img * h_img):
               print("🔥 BBOX ERROR:", (x, y, w, h))
               continue

           cv2.rectangle(
              bbox_img,
              (int(x), int(y)),
              (int(x + w), int(y + h)),
              color,
              2
           )
           cv2.putText(
              bbox_img,
              class_name,
              (int(x), int(y-5)),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              color,
              1
           )

        # 存bbox 2d圖
        bbox_2d_file_name = f"bbox_2d_frame_{timestamp}.png"
        bbox_2d_dir_name = f"data_storage/bbox_2d_image/{self.dataset.time_str}"
        bbox_2d_path = f"{bbox_2d_dir_name}/{bbox_2d_file_name}"
        os.makedirs(bbox_2d_dir_name, exist_ok=True)
        cv2.imwrite(bbox_2d_path, bbox_img)


    except Exception as e:
        print("🔥 RGB ERROR:", e)

  # 處理instance segmentation 
  def process_inst_seg(self, inst, sem, timestamp):
    color_vis = utils.colorize_mask(inst)

    height, width = inst.shape
    file_name = f"ins_frame_{timestamp}.png"
    ins_seg_image_path = f"data_storage/ins_seg_image/{self.dataset.time_str}/{file_name}"
    cv2.imwrite(ins_seg_image_path, color_vis)
    print(ins_seg_image_path)

    date_capture = datetime.datetime.fromtimestamp(
    os.path.getctime(ins_seg_image_path)
      ).strftime("%Y-%m-%d %H:%M:%S")

    image_id = self.dataset.add_image(file_name, width, height, date_capture)
    instances = self.converter.extract_instances(inst)
    inst_to_class = self.build_inst_to_class(inst, sem)
    merged_instances = self.merge_instances_by_class(instances, inst_to_class, sem)
    
    for class_name, mask in merged_instances:
        category_id = self.dataset.add_category(class_name)
        
        segmentation  = self.converter.mask_to_polygon(mask)
        bbox = self.converter.mask_to_bbox(mask)
        h_img, w_img = mask.shape

        if bbox[2] * bbox[3] >= 0.95 * (w_img * h_img):
            print("🔥 BBOX ERROR:", bbox)
            continue
        
        area = self.converter.mask_area(mask)
        self.dataset.add_annotation(
           image_id,
           category_id,
           segmentation,
           bbox,
           area
        )

  def build_inst_to_class(self, inst, sem):
     mapping = {}
     inst_ids = np.unique(inst)

     for inst_id in inst_ids:
        if inst_id == 0:
           continue
        
        mask = (inst == inst_id)
        sem_pixels = sem[mask].astype(np.int64)

        if len(sem_pixels) == 0:
           continue
        
        sem_id = np.bincount(sem_pixels).argmax()

        if sem_id == 0:
           continue
        
        class_name = self.semantic_id_to_name.get(sem_id, "unknown")
        mapping[inst_id] = class_name
        print(f"inst={inst_id} → sem={sem_id} → class={class_name}")

     return mapping
  
  def merge_instances_by_class(self, inst, inst_to_class, sem):
      merged = {}

      for inst_id, mask in inst:
          inst_id = int(inst_id)
          class_name = inst_to_class.get(inst_id, "unknown")

          # 過濾垃圾訊息
          if class_name in ["unknown", "unlabelled", "background"]:
              continue
          
          clean_mask = mask.copy().astype(bool)

          sem_pixels = sem[clean_mask].astype(np.int64)
          if len(sem_pixels) == 0:
              continue
          sem_id = np.bincount(sem_pixels).argmax()
          clean_mask = clean_mask & (sem == sem_id)
          
          # 將同一類別的 instance mask 合併
          if class_name not in merged:
              merged[class_name] = clean_mask.copy()

          else:
              merged[class_name] = np.logical_or(merged[class_name], clean_mask)
      
      result = []
      for class_name, merged_mask in merged.items():
          result.append((class_name, merged_mask.astype(np.uint8)))

      return result    
 
  def semantic_callback(self, msg):
      data = json.loads(msg.data)
      new_map = {}

      for k, v in data.items():
         if not k.isdigit():
               continue

         class_id = int(k)

         # 格式 A: {'hospital_bed_01': 'hospital_bed'}
         if isinstance(v, dict):
               # 排除 BACKGROUND / UNLABELLED 等保留字
               inner_val = list(v.values())[0]
               if isinstance(inner_val, str) and inner_val.upper() not in ("BACKGROUND", "UNLABELLED"):
                  new_map[class_id] = inner_val.lower()

         # 格式 C: "curtain_01:curtain"
         elif isinstance(v, str) and ":" in v:
               class_name = v.split(":")[-1].strip().lower()
               if class_name.upper() not in ("BACKGROUND", "UNLABELLED"):
                  new_map[class_id] = class_name

         # 格式 D: "/World/IVPole_0302" → 跳過，無法解析
         else:
               pass

      if new_map:
         # 只用「更好的資料」更新，不允許覆蓋已有的有效 label
         for k, v in new_map.items():
               if k not in self.semantic_id_to_name:
                  self.semantic_id_to_name[k] = v
                  print(f"  ✅ 新增 sem_id={k} → {v}")
               else:
                  print(f"  ⏩ 保留 sem_id={k} → {self.semantic_id_to_name[k]}（忽略新值 {v}）")
      else:
         print("⚠️ 忽略無效 semantic_labels")

      print(f"  當前 semantic map: {self.semantic_id_to_name}")

  def destroy_node(self):
     self.dataset.save_json()
     super().destroy_node()

def main():
   rclpy.init()
   node = DatasetExporter()

   try:
      rclpy.spin(node)

   except KeyboardInterrupt:
      pass
   
   finally:
      if node is not None:
        try:
            node.destroy_node()
        except Exception as e:
            print("Node already destroyed:", e)

      try:
          rclpy.shutdown()
      except Exception as e:
          print("ROS2 context already shutdown:", e)
    
if __name__ == "__main__":
  main()


