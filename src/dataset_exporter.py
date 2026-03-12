import datetime
import os
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from dataset_manager import DatasetManager
from coco_converter import CocoConverter


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

    now = datetime.datetime.now()
    self.time_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    # 建立影像訂閱
    self.subscription = self.create_subscription(
      Image,
      "/instance_segmentation",
      self.listener_callback,
      10
    )

  def listener_callback(self, msg):

    if time.time() - self.start_time > 30:
      print("Reached 10 seconds, shutting down...")
      rclpy.shutdown()
      return
  
    image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    vis = image.astype(np.float32)

    if vis.max() > 0:
       vis = vis / vis.max()*255
    vis = vis.astype(np.uint8)

    height, width = image.shape
    file_name = f"frame_{msg.header.stamp.sec}.png"
    image_path = f"instance_data_storage/image/{self.dataset.time_str}/{file_name}"
    cv2.imwrite(image_path, vis)
    print(image_path)

    date_capture = datetime.datetime.fromtimestamp(
    os.path.getctime(image_path)
      ).strftime("%Y-%m-%d %H:%M:%S")
    print(date_capture)
    image_id = self.dataset.add_image(file_name, width, height, date_capture)
    instances = self.converter.extract_instances(image)
    
    for inst_id, mask in instances:
        segmentation  = self.converter.mask_to_polygon(mask)
        bbox = self.converter.mask_to_bbox(mask)
        area = self.converter.mask_area(mask)
        category_id = 1
        self.dataset.add_annotation(
           image_id,
           category_id,
           segmentation,
           bbox,
           area
        )

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


