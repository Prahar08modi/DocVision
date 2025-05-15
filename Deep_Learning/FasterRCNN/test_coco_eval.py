from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json, argparse

def test_coco_eval(ann_file, subset_size=None):
    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    if subset_size:
        img_ids = img_ids[:subset_size]

    # build perfect “predictions”
    dummy = []
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        for ann in coco_gt.loadAnns(ann_ids):
            dummy.append({
              "image_id":    img_id,
              "category_id": ann["category_id"],
              "bbox":        ann["bbox"],
              "score":       1.0
            })

    with open("dummy_results.json","w") as f:
        json.dump(dummy, f)

    # run COCOeval, restrict to our img_ids and set full maxDets
    coco_dt   = coco_gt.loadRes("dummy_results.json")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    coco_eval.params.imgIds   = img_ids
    coco_eval.params.maxDets  = [1, 10, 1000]   # now length=3!

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-file",    required=True)
    p.add_argument("--subset-size", type=int, default=100)
    args = p.parse_args()
    test_coco_eval(args.ann_file, args.subset_size)
