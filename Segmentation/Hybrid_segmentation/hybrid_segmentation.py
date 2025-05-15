#!/usr/bin/env python3
# hybrid_segmentation.py

import os
import json
import argparse
from tqdm import tqdm

import cv2
import numpy as np

import os, sys

# ─── add our sibling folders to the import path ──────────────────
# assume hybrid_segmentation.py lives in ./Hybrid_segmentation/
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DOCSTRUM_DIR = os.path.join(PROJECT_ROOT, "Docstrum")
XYCUT_DIR   = os.path.join(PROJECT_ROOT, "Recursive_xycut")
sys.path.insert(0, DOCSTRUM_DIR)
sys.path.insert(0, XYCUT_DIR)

# ─── XY‐Cut imports ─────────────────────────────────────
from process_xycut import recursive_xy_cut_debug

# ─── DocStrum + dynamic morphology imports ────────────
from docstrum_components import get_connected_components
from docstrum_geometry   import build_nn_graph, estimate_page_orientation, \
                               estimate_stroke_height, estimate_char_spacing
from docstrum_lines      import build_docstrum_lines, clusters_to_boxes
from docstrum_dynamic_morph_blocks import dynamic_morphological_merge

def hybrid_segment_page(
    gray: np.ndarray,
    binary: np.ndarray,
    args
):
    H, W = binary.shape
    final_boxes = []

    # 1) coarse XY‑Cut blocks
    xy_blocks = recursive_xy_cut_debug(
        binary,
        min_block=args.xy_min_block,
        thr_h=args.xy_thr_h, thr_v=args.xy_thr_v,
        gap_h=args.xy_gap_h, gap_v=args.xy_gap_v,
        k_h=args.xy_k_h, k_v=args.xy_k_v
    )

    # 2) for each block, decide text vs non‑text
    for (x0, y0, bw, bh) in xy_blocks:
        sub_bin = binary[y0:y0+bh, x0:x0+bw]
        # count CCs inside this block
        comps = get_connected_components(sub_bin,
                                         min_area=args.cc_min_area,
                                         max_area=args.cc_max_area)
        if len(comps) >= args.text_cc_thresh:
            # ── text block: apply DocStrum + dynamic morphology ──
            # 2a) geometry estimates
            nn_graph = build_nn_graph(comps, k=args.k)
            if not nn_graph:
                # too few CCs: fallback to whole block
                final_boxes.append((x0, y0, bw, bh))
                continue
            ori     = estimate_page_orientation(nn_graph)
            stroke  = estimate_stroke_height(comps)
            spacing = estimate_char_spacing(nn_graph, ori)

            # 2b) DocStrum line clustering
            clusters = build_docstrum_lines(
                comps, ori, spacing, stroke,
                min_dist_factor=args.min_dist_factor,
                max_dist_factor=args.max_dist_factor,
                angle_tol=args.angle_tol
            )
            line_boxes = clusters_to_boxes(clusters, comps)

            # 2c) dynamic morphological merge → paragraph boxes
            para_boxes = dynamic_morphological_merge(
                (bh, bw),
                # note: dynamic function expects coords relative to (0,0)
                line_boxes,
                stroke_height=stroke,
                dynamic_pct=args.dynamic_pct,
                open_factor=args.open_factor
            )

            # offset back to original page coords
            for (x, y, w, h) in para_boxes:
                final_boxes.append((x0 + x, y0 + y, w, h))
        else:
            # ── non‑text: keep coarse XY block
            final_boxes.append((x0, y0, bw, bh))

    return final_boxes

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid: XY‑Cut coarse + DocStrum dynamic morph for text blocks"
    )
    parser.add_argument("--set-json", help="Path to COCO JSON listing only the subset to process")
    parser.add_argument("--input-dir",  required=True,
                        help="Folder of page PNGs")
    parser.add_argument("--output-coco",required=True,
                        help="Output COCO JSON for hybrid blocks")

    # XY‑Cut hyper‑params (match your process_xycut.py)
    parser.add_argument("--xy-min-block", type=int,   default=70)
    parser.add_argument("--xy-thr-h",     type=float, default=0.2)
    parser.add_argument("--xy-thr-v",     type=float, default=0.05)
    parser.add_argument("--xy-gap-h",     type=int,   default=10)
    parser.add_argument("--xy-gap-v",     type=int,   default=7)
    parser.add_argument("--xy-k-h",       type=int,   default=25)
    parser.add_argument("--xy-k-v",       type=int,   default=15)

    # CC filter inside blocks
    parser.add_argument("--cc-min-area", type=int, default=10)
    parser.add_argument("--cc-max-area", type=int, default=5000)
    # threshold to decide “text‐heavy” block
    parser.add_argument("--text-cc-thresh", type=int, default=200,
                        help="min #CCs in XY block to treat as text")

    # DocStrum hyper‑params
    parser.add_argument("--k",               type=int,   default=5)
    parser.add_argument("--angle-tol",       type=float, default=30.0)
    parser.add_argument("--min-dist-factor", type=float, default=0.5)
    parser.add_argument("--max-dist-factor", type=float, default=6.0)

    # dynamic morphological merge
    parser.add_argument("--dynamic-pct", type=float, default=70.0)
    parser.add_argument("--open-factor", type=float, default=0.4)

    args = parser.parse_args()
    valid_fnames = None
    # If provided, load the subset JSON and build GT filename→ID map
    fname2id = {}
    if args.set_json:
        with open(args.set_json, "r") as f:
            subset = json.load(f)
        # map file_name → original image_id
        valid_fnames = {img["file_name"] for img in subset["images"]}
        fname2id = { img["file_name"]: img["id"]
                     for img in subset["images"] }
        print(f"▶ Restricting to {len(fname2id)} files from {args.set_json}")

    coco = {
        "info": {}, "licenses": [], "images": [], "annotations": [],
        "categories": [{"id":1,"name":"block","supercategory":"layout"}]
    }
    ann_id = 1

    # list everything in the dir...
    all_files = sorted(os.listdir(args.input_dir))
    # ...keep only images...
    all_files = [f for f in all_files if f.lower().endswith((".png",".jpg",".jpeg"))]
    # ...and if set-json given, restrict to that subset
    if valid_fnames is not None:
        all_files = [f for f in all_files if f in valid_fnames]
        print(f"▶ Processing {len(all_files)} images after filtering")

    for fname in tqdm(all_files, desc="Hybrid seg pages"):
        img_path = os.path.join(args.input_dir, fname)
        gray  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        # always inverted Otsu
        _, bin_img = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        boxes = hybrid_segment_page(gray, bin_img, args)

        # record image
        H, W = gray.shape
        # look up the original image_id
        if fname not in fname2id:
            # skip any images not listed in the set JSON
            continue
        gt_img_id = fname2id[fname]
        coco["images"].append({
            "id":        int(gt_img_id),
            "width":     int(W),
            "height":    int(H),
            "file_name": fname
        })

        # record annotations
        for (x, y, w, h) in boxes:
            coco["annotations"].append({
                "id":          ann_id,
                "image_id":    int(gt_img_id),
                "category_id": 1,
                "bbox":        [int(x), int(y), int(w), int(h)],
                "area":        int(w*h),
                "iscrowd":     0,
                "segmentation":[[int(x),int(y),
                                 int(x+w),int(y),
                                 int(x+w),int(y+h),
                                 int(x),int(y+h)]]
            })
            ann_id += 1

    # write out
    with open(args.output_coco, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"\n✅ Hybrid segmentation done: "
          f"{len(coco['images'])} images, {len(coco['annotations'])} blocks")

if __name__ == "__main__":
    main()
