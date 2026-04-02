"""找出instance mask，將原先的數字矩陣，轉換成coco segmentation"""

import numpy as np
from skimage import measure

class CocoConverter:
  def extract_instances(self, segmentation_image):
    """
    找出所有 instance_id
    """
    instances_ids = np.unique(segmentation_image)
    instances = []

    for inst_id in instances_ids:
      if inst_id ==0: # 背景
        continue
      mask = segmentation_image == inst_id

      instances.append((inst_id, mask))

    return instances
  
  def mask_to_polygon(mask):
      # 先補一圈背景
      padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

      contours = measure.find_contours(padded, 0.5)
      segmentation = []

      for contour in contours:
          contour = np.flip(contour, axis=1)   # yx -> xy
          contour -= 1                         # 因為 pad 了，要扣回來
          segmentation.append(contour.ravel().tolist())

    return segmentation 
  def mask_to_bbox(self, mask):
    y_indices, x_indices = np.where(mask)

    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))

    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))

    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]
  
  def mask_area(self, mask):
    """計算影像中單一物件的面積"""
    return int(np.sum(mask))