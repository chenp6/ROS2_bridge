import datetime
import json
import os
import fixed_categories

class DatasetManager:
  def __init__(self, dataset_dir="data_storage", time_stamp = ""):
    self.time_str = time_stamp
    self.dataset_dir = dataset_dir
    self.rgb_image_dir = os.path.join(self.dataset_dir, "rgb_image")
    self.instance_image_dir = os.path.join(self.dataset_dir, "ins_seg_image")
    self.bbox_2d_image_dir = os.path.join(self.dataset_dir, "bbox_2d_image")
    self.instance_annotation_dir = os.path.join(dataset_dir, "annotations")
    self.instance_date_dir = os.path.join(self.instance_image_dir,self.time_str)
    
    os.makedirs(self.instance_date_dir, exist_ok = True)
    os.makedirs(self.rgb_image_dir, exist_ok = True)
    os.makedirs(self.instance_image_dir, exist_ok=True)
    os.makedirs(self.bbox_2d_image_dir, exist_ok = True)
    os.makedirs(self.instance_annotation_dir, exist_ok=True)

    self.coco = {
      "images":[],
      "annotations":[],
      "categories":[]
    }

    self.category_id_counter = 500
    self.image_id = 0
    self.annotation_id = 0
    self.category_map = {}

  def add_category(self, name):
    """
      新增一個category
    """
    if name in self.category_map:
       return self.category_map[name]
    
    if name in fixed_categories.FIXED_CATEGORIES:
       cid = fixed_categories.FIXED_CATEGORIES[name]
    else:
       self.category_id_counter += 1
       cid = self.category_id_counter
       
    self.category_map[name] = cid
    
    self.coco["categories"].append({
      "id":cid,
      "name":name
    })
    print (self.category_map)
    return cid

  def add_image(self, file_name, width, height, date_capture):
    """
      image record
    """
    self.image_id +=1

    image_info = {
      "id": self.image_id,
      "file_name":file_name,
      "width":width,
      "height":height,
      "date_captured": date_capture
    }
    self.coco["images"].append(image_info)

    return self.image_id
  
  def add_annotation(self, image_id, category_id, segmentation, bbox, area):
    self.annotation_id += 1

    #print(self.annotation_id)
    ann = {
      "id": self.annotation_id,
      "image_id": image_id,
      "category_id": category_id,
      "segmentation": segmentation,
      "bbox": bbox,
      "area": area,
      "iscrowd":0,
    }

    self.coco["annotations"].append(ann)

  def save_json(self):
    output_file = os.path.join(
      self.instance_annotation_dir,
      f"instances_{self.time_str}.json"
    )

    with open(output_file, "w") as f:
            json.dump(self.coco, f, indent=4)

    print("COCO JSON saved:", output_file)