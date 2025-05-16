import json
import pandas as pd

def load_coco_annotations(json_path):
    """
    Load a COCO-format annotation file and return a DataFrame
    with columns: image_id, file_name, bbox, category.
    
    Args:
        json_path (str): Path to COCO JSON (e.g. train.json)
    
    Returns:
        pd.DataFrame: One row per annotation with these columns:
            - image_id   : integer image ID
            - file_name  : string, e.g. "000001.png"
            - bbox       : [x, y, width, height]
            - category   : human-readable class name
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # build lookup tables
    images = {img['id']: img['file_name'] for img in data['images']}
    categories = {cat['id']: cat['name']       for cat in data['categories']}
    
    # flatten annotations
    records = []
    for ann in data['annotations']:
        img_id = ann['image_id']
        cid    = ann.get('category_id', None)
        category_name = categories.get(cid, None)
        records.append({
            'image_id': img_id,
            'file_name': images[img_id],
            'bbox': ann['bbox'],
            'category': category_name
        })
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    # quick sanity check
    df = load_coco_annotations('Data/DocLayNet_core/COCO/train.json')
    print("Loaded annotations:", len(df))
    print(df.head())
