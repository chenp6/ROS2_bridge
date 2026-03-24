
import numpy as np
import cv2

color_map = {}

# 根據圖片中不同的數字矩陣傳回不同顏色
def colorize_mask (mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
  
    unique_ids = np.unique(mask)
    for uid in unique_ids:
        if uid == 0:
          continue

        color_val = get_color(uid)  
        color[mask == uid] = color_val

    return color

# 藉由重現不同的inst_id 顏色（局部亂數產生器）
def get_color(inst_id):
    np.random.seed(inst_id)   # 🔥 保證跨版本一致
    color = np.random.randint(0, 255, 3)
    return tuple(int(c) for c in color)

