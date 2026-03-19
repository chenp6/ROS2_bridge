import datetime
import os
import time
import json

import rclpy
from rclpy.node import Node

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

    self.bridge = CvBridge()
    self.converter = CocoConverter()
    self.dataset = DatasetManager(time_stamp=self.time_str)
    self.semantic_map = {}
    self.latest_rgb = None
    self.latest_seg = None

    self.start_time = time.time()

    # semantic 訂閱
    self.semantic_sub = self.create_subscription(
       String,
       "/semantic_labels",
       self.semantic_callback,
       10
    )

    # 建立instance_seg影像訂閱
    self.inst_seg_sub = self.create_subscription(
      Image,
      "/instance_segmentation",
      self.inst_seg_callback,
      10
    )

    # 建立rgb 影像訂閱
    self.rgb_sub = self.create_subscription(
            Image,
            "/rgb",
            self.rgb_callback,
            10
         )
    
  def rgb_callback(self, msg):
     print("RGB received")
     self.latest_rgb = msg
     self.try_process()

  def inst_seg_callback(self, msg):
    print("SEG received")
    self.latest_seg = msg
    self.try_process()

  # 同步資料處理
  def try_process(self):
    print("RGB:", self.latest_rgb is not None, 
      "SEG:", self.latest_seg is not None)
    
    if self.latest_rgb is None or self.latest_seg is None:
       return
    
    if time.time() - self.start_time > 30:
      print("Reached 30 seconds, shutting down...")
      rclpy.shutdown()
      return
    
    rgb_msg = self.latest_rgb
    seg_msg = self.latest_seg
    timestamp = rgb_msg.header.stamp.sec

    self.process_pair(rgb_msg, seg_msg, timestamp)
    self.process_inst_seg(seg_msg, timestamp)

    self.latest_rgb = None
    self.latest_seg = None

  # 處理 rgb, bbox2d
  def process_pair(self, rgb_msg, seg_msg, timestamp):
    try:
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
        seg = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding="passthrough")
        instance = self.converter.extract_instances(seg)

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

        rgb_success = cv2.imwrite(rgb_image_path, rgb)
        print(f"rgb saved: {rgb_success} → {rgb_image_path}")

        # 畫 bbox
        bbox_img = rgb.copy()

        for inst_id, mask in instance:
           color = utils.get_color(inst_id)
           color = tuple(int(c) for c in color)
           x, y, w, h = self.converter.mask_to_bbox(mask)

           cv2.rectangle(
              bbox_img,
              (int(x), int(y)),
              (int(x + w), int(y + h)),
              color,
              2
           )

           label = self.semantic_map.get(inst_id, str(inst_id))
           cv2.putText(
              bbox_img,
              label,
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
        bbox_2d_sucess = cv2.imwrite(bbox_2d_path, bbox_img)


    except Exception as e:
        print("🔥 RGB ERROR:", e)

  # 處理instance segmentation 
  def process_inst_seg(self, seg_msg, timestamp):
    seg = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding="passthrough")
    color_vis = utils.colorize_mask(seg)

    height, width = seg.shape
    file_name = f"ins_frame_{timestamp}.png"
    ins_seg_image_path = f"data_storage/ins_seg_image/{self.dataset.time_str}/{file_name}"
    cv2.imwrite(ins_seg_image_path, color_vis)
    print(ins_seg_image_path)

    date_capture = datetime.datetime.fromtimestamp(
    os.path.getctime(ins_seg_image_path)
      ).strftime("%Y-%m-%d %H:%M:%S")
    print(date_capture)
    image_id = self.dataset.add_image(file_name, width, height, date_capture)
    instances = self.converter.extract_instances(seg)
    
    for inst_id, mask in instances:
        category_name = self.semantic_map.get(inst_id, "unknown")
        category_id = self.dataset.add_category(category_name)
        
        segmentation  = self.converter.mask_to_polygon(mask)
        bbox = self.converter.mask_to_bbox(mask)
        area = self.converter.mask_area(mask)
        self.dataset.add_annotation(
           image_id,
           category_id,
           segmentation,
           bbox,
           area
        )


   # 偵測檔案中全部標籤
  def semantic_callback(self, msg):
    data = json.loads(msg.data)

    for k, v in data.items():
        if not k.isdigit():
            continue

        inst_id = int(k)

        # ✅ 只收 dict（最乾淨）
        if isinstance(v, dict):
            class_name = list(v.values())[0]
            self.semantic_map[inst_id] = class_name

        # ⚠️ 如果是 "xxx:yyy" 才接受
        elif isinstance(v, str) and ":" in v:
            class_name = v.split(":")[-1]
            self.semantic_map[inst_id] = class_name

        # ❌ 完全忽略 prim path
        else:
            continue

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


