import json
import cv2
import os

def draw_coco_bboxes(coco_json_path, image_dir, output_dir):
    """
    coco_json_path: path to COCO-format annotations (dict with 'images' and 'annotations')
    image_dir: directory containing the images
    output_dir: where to save the images with drawn boxes
    """
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Build a mapping from image_id to filename
    img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group annotations by image_id
    anns_by_image = {}
    for ann in coco['annotations']:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    # Process each image
    for img_id, filename in img_id_to_filename.items():
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        # Draw each bbox
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann['bbox']  # COCO format: [top-left x, top-left y, width, height]
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=2)

            # Optionally, draw the category id
            cat_id = ann.get('category_id', None)
            if cat_id is not None:
                cv2.putText(
                    img, str(cat_id), 
                    (pt1[0], pt1[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1
                )

        # Show and save
        cv2.imshow('bboxes', img)
        cv2.waitKey(1)  # press any key in the image window to move to next
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    draw_coco_bboxes(
        coco_json_path="/Users/praharmodi/Study/Sem4/MSML640/Project/DocVision/Segmentation/New_style/Recursive_xycut/block_outputs/0a2c475664e7e18068424c8d29e5d819158494fa9b20bf3375cf31b3f17d93cd_blocks_coco.json",
        image_dir="/Users/praharmodi/Study/Sem4/MSML640/Project/DocVision/Segmentation/New_style/test_images",
        output_dir="images_with_bboxes/"
    )
