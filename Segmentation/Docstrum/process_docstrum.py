#!/usr/bin/env python3
# process_val_dynamic_morph_coco.py

import os
import json
import random
import argparse
from tqdm import tqdm

import cv2
import numpy as np

# ── Import your existing modules ──
from Segmentation.Docstrum.docstrum_components import get_connected_components
from Segmentation.Docstrum.docstrum_geometry   import build_nn_graph, estimate_page_orientation, \
                               estimate_stroke_height, estimate_char_spacing
from Segmentation.Docstrum.docstrum_lines      import build_docstrum_lines, clusters_to_boxes
from Segmentation.Docstrum.docstrum_dynamic_morph_blocks import dynamic_morphological_merge

def main():
    p = argparse.ArgumentParser(
        description="Process a random subset of val.json images via dynamic morph merge"
    )
    p.add_argument("--input-dir",   required=True,
                   help="Folder containing all PNGs")
    p.add_argument("--val-json",    required=True,
                   help="COCO JSON listing validation images")
    p.add_argument("--output-coco", required=True,
                   help="Path to write output COCO JSON")
    p.add_argument("--sample-size", type=int, default=1000,
                   help="How many val images to process")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for sampling")

    # pipeline hyperparams
    p.add_argument("--min-area",       type=int,   default=10)
    p.add_argument("--max-area",       type=int,   default=5000)
    p.add_argument("--k",              type=int,   default=5)
    p.add_argument("--angle-tol",      type=float, default=30.0)
    p.add_argument("--min-dist-factor",type=float, default=0.5)
    p.add_argument("--max-dist-factor",type=float, default=6.0)
    p.add_argument("--dynamic-pct",    type=float, default=80.0)
    p.add_argument("--open-factor",    type=float, default=0.2)
    args = p.parse_args()

    # 1. Load val.json, extract filenames & build GT ID map
    with open(args.val_json, "r") as f:
        val = json.load(f)
    all_val_files = [img["file_name"] for img in val["images"]]
    fname2id      = {img["file_name"]: img["id"] for img in val["images"]}

    # 2. Sample
    random.seed(args.seed)
    sample = random.sample(all_val_files,
                           min(args.sample_size, len(all_val_files)))
    print(f"Sampling {len(sample)}/{len(all_val_files)} val images (seed={args.seed})")

    # 3. Prepare COCO structure
    coco = {
        "info":       val.get("info", {}),
        "licenses":   val.get("licenses", []),
        "images":     [],
        "annotations":[],
        "categories":[{"id":1,"name":"paragraph","supercategory":"layout"}]
    }
    ann_id = 1

    # 4. Process each image
    for fname in tqdm(all_val_files, desc="Processing val pages"):
        img_path = os.path.join(args.input_dir, fname)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            tqdm.write(f"⚠️  Cannot load {fname}, skipping")
            continue
        H, W = gray.shape

        # binarize
        _, bin_img = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # CC → components
        comps = get_connected_components(bin_img,
                                         min_area=args.min_area,
                                         max_area=args.max_area)
        if len(comps) < 2:
            tqdm.write(f"⚠️  Too few components in {fname}, skipping")
            continue

        # geometry
        nn      = build_nn_graph(comps, k=args.k)
        ori     = estimate_page_orientation(nn)
        stroke  = estimate_stroke_height(comps)
        spacing = estimate_char_spacing(nn, ori)

        # DocStrum lines → line_boxes
        clusters   = build_docstrum_lines(comps, ori, spacing, stroke,
                                          min_dist_factor=args.min_dist_factor,
                                          max_dist_factor=args.max_dist_factor,
                                          angle_tol=args.angle_tol)
        line_boxes = clusters_to_boxes(clusters, comps)

        # Dynamic morphological merge → para_boxes
        para_boxes = dynamic_morphological_merge(
            (H, W), line_boxes, stroke,
            dynamic_pct=args.dynamic_pct,
            open_factor=args.open_factor
        )

        # 5. Append image entry with GT ID
        gt_img_id = fname2id[fname]
        coco["images"].append({
            "id":        gt_img_id,
            "width":     W,
            "height":    H,
            "file_name": fname
        })

        # 6. Append each proposal annotation under the GT image ID
        for x, y, w, h in para_boxes:
            ix, iy, iw, ih = int(x), int(y), int(w), int(h)
            coco["annotations"].append({
                "id":          ann_id,
                "image_id":    gt_img_id,
                "category_id": None,
                "bbox":        [ix, iy, iw, ih],
                "area":        iw * ih,
                "iscrowd":     0,
                "segmentation":[[ix, iy,
                                 ix+iw, iy,
                                 ix+iw, iy+ih,
                                 ix, iy+ih]]
            })
            ann_id += 1

    # 7. Write output
    with open(args.output_coco, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"\n✅ Done! Saved {len(coco['images'])} images and "
          f"{len(coco['annotations'])} annotations to {args.output_coco}")

if __name__ == "__main__":
    main()
