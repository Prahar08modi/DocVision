#!/usr/bin/env python3
import os
import json
import argparse
import cv2

def visualize_coco_boxes(input_dir, coco_json, out_dir=None):
    # Load COCO JSON
    with open(coco_json, 'r') as f:
        coco = json.load(f)

    # Build a map from image_id to its annotations
    anns_by_img = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        anns_by_img.setdefault(img_id, []).append(ann)

    # Build a map from image_id to filename
    img_info = {img['id']: img for img in coco['images']}

    # Process each image in the JSON
    for img_id, info in img_info.items():
        fname = info['file_name']
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Could not load image: {img_path}")
            continue

        # Draw each bbox
        for ann in anns_by_img.get(img_id, []):
            x, y, w, h = ann['bbox']
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

        # Display or save
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, fname)
            cv2.imwrite(save_path, img)
        else:
            cv2.imshow(fname, img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            cv2.destroyWindow(fname)

    if not out_dir:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize COCO-format block proposals on images"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Folder containing your test images"
    )
    parser.add_argument(
        "--coco-json", required=True,
        help="COCO JSON file with block annotations"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="If provided, save overlays here instead of displaying"
    )
    args = parser.parse_args()

    visualize_coco_boxes(args.input_dir, args.coco_json, args.out_dir)
