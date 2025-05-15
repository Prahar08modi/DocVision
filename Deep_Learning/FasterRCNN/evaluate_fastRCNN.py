import argparse
import json
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class COCODataset(Dataset):
    def __init__(self, img_folder, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_folder = img_folder
        self.transform = transform or transforms.ToTensor()

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = f"{self.img_folder}/{info['file_name']}"
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        return img_id, img_tensor

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return list(zip(*batch))

def run_evaluation(model, data_loader, coco_gt, device):
    model.eval()
    results = []

    total_batches = len(data_loader)
    print(f"⏳ Starting inference over {total_batches} batches…")
    with torch.no_grad():
        for batch_idx, (img_ids, imgs) in enumerate(tqdm(data_loader, desc="Infer", unit="batch")):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for img_id, output in zip(img_ids, outputs):
                boxes = output['boxes'].cpu().tolist()
                scores = output['scores'].cpu().tolist()
                labels = output['labels'].cpu().tolist()
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    results.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, w, h],
                        "score": float(score)
                    })

    # ─── DEBUG: how many detections did we actually get?
    print(f"ℹ️  Collected {len(results)} total detections.")
    if len(results) == 0:
        print("⚠️  No detections to evaluate!  Skipping COCOeval.  "
              "Check that your model loaded correctly and is producing outputs.")
        return

    # write out detections
    with open("results.json", "w") as f:
        json.dump(results, f)

    # load detections and run COCOeval
    coco_dt = coco_gt.loadRes("results.json")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load your trained model ---
    # Example: torchvision’s Fast R-CNN backbone
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        num_classes=args.num_classes,
        box_score_thresh=0.0,        # keep *every* detection
        box_nms_thresh=0.5,
        box_detections_per_img=1000   # allow up to 1k boxes per image
    )

    print(f"▶️ score_thresh = {model.roi_heads.score_thresh:,}")
    print(f"▶️ detections_per_img = {model.roi_heads.detections_per_img:,}")

    # --- load checkpoint robustly with debug info ---
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # since you saved only state_dict, it should be a dict of tensors
    if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError("Unexpected checkpoint format")

    # load with strict=False so we can see missing/unexpected keys
    load_info = model.load_state_dict(state_dict, strict=False)
    print("🔑 Missing keys:", load_info.missing_keys)
    print("⚠️ Unexpected keys:", load_info.unexpected_keys)
    model.to(device)

    # ─── override Faster R-CNN post‑processing ───
    model.roi_heads.score_thresh       = 0.0
    model.roi_heads.detections_per_img = 1000

    # --- prepare dataset ---
    dataset = COCODataset(
        img_folder=args.image_dir,
        ann_file=args.ann_file,
        transform=transforms.ToTensor()
    )

    model.eval()
    # grab one sample
    img_id, img = dataset[0]
    with torch.no_grad():
        out = model([img.to(device)])
    print("📦 Single-image raw output:", {
        "boxes": out[0]["boxes"].shape,
        "scores": out[0]["scores"][:5],
        "labels": out[0]["labels"][:5]
    })

    # build a COCO‑pretrained baseline with the same overrides
    baseline = fasterrcnn_resnet50_fpn(
        pretrained=True,
        box_score_thresh=0.0,
        box_nms_thresh=0.5,
        box_detections_per_img=1000
    ).eval().to(device)

    with torch.no_grad():
        base_out = baseline([img.to(device)])[0]

    print("🚀 Baseline pretrained boxes:", base_out["boxes"].shape,
        "scores:", base_out["scores"][:5])

    # test on one train image
    train_dataset = COCODataset(
        img_folder=args.image_dir,
        ann_file=args.ann_file.replace("val.json", "train.json"),
        transform=transforms.ToTensor()
    )
    train_img_id, train_img = train_dataset[0]
    with torch.no_grad():
        train_out = model([train_img.to(device)])[0]
    print("🏋️ Train‐set boxes:", train_out["boxes"].shape,
        "scores:", train_out["scores"][:5])


    # >>> SUBSET HACK: only keep the first args.subset_size imgs
    if args.subset_size is not None:
        print(f"⚡ Running on a subset of {args.subset_size} images (of {len(dataset)} total)")
        dataset.ids = dataset.ids[: args.subset_size]

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=4
    )
    coco_gt = dataset.coco  # ground‑truth COCO API

    # --- run eval ---
    run_evaluation(model, data_loader, coco_gt, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fast R-CNN on COCO-format data")
    parser.add_argument("--checkpoint", required=True,
                        help="path to your trained model checkpoint (.pth)")
    parser.add_argument("--image-dir", required=True,
                        help="directory with val images")
    parser.add_argument("--ann-file", required=True,
                        help="path to val.json annotation file")
    parser.add_argument("--num-classes", type=int, required=True,
                        help="number of classes (including background)")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="if set, only evaluate on the first N images of the val set")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="batch size for inference")
    args = parser.parse_args()
    main(args)
