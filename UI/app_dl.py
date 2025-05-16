# app_dl.py
import os
import sys
import json
from types import SimpleNamespace

# ─── turn off interactive plt.show() ────────────────────────────────
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib

# ─── Deep‑Learning imports ──────────────────────────────────────────
import torch
import torchvision
from torchvision.transforms import functional as F

# ─── add project root to path ─────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── segmentation imports ───────────────────────────────────────────
from Segmentation.Recursive_xycut.process_xycut import recursive_xy_cut_debug
from Segmentation.Docstrum.docstrum_components import get_connected_components
from Segmentation.Docstrum.docstrum_geometry import (
    build_nn_graph, estimate_page_orientation,
    estimate_stroke_height, estimate_char_spacing
)
from Segmentation.Docstrum.docstrum_lines import build_docstrum_lines, clusters_to_boxes
from Segmentation.Docstrum.docstrum_dynamic_morph_blocks import dynamic_morphological_merge
from Segmentation.Hybrid_segmentation.hybrid_segmentation import hybrid_segment_page
from Deep_Learning.model_loader import load_fastrcnn_model

# ─── full COCO ID→name map (DocLayNet categories) ────────────────────
CATEGORY_MAP = {
    1:  "Caption",
    2:  "Footnote",
    3:  "Formula",
    4:  "List-item",
    5:  "Page-footer",
    6:  "Page-header",
    7:  "Picture",
    8:  "Section-header",
    9:  "Table",
    10: "Text",
    11: "Title"
}

# ─── hyperparameters ────────────────────────────────────────────────
seg_args = SimpleNamespace(
    xy_min_block=70, xy_thr_h=0.2, xy_thr_v=0.05,
    xy_gap_h=10, xy_gap_v=7,
    xy_k_h=25, xy_k_v=15,
    cc_min_area=10, cc_max_area=5000, text_cc_thresh=200,
    k=5, angle_tol=30.0,
    min_dist_factor=0.5, max_dist_factor=6.0,
    dynamic_pct=70.0, open_factor=0.4
)

# ─── model paths ───────────────────────────────────────────────────
PKL_PATH  = os.path.join(ROOT,    "Classification", "lightgbm_doclaynet.pkl")

# ─── segmentation helpers ──────────────────────────────────────────
def docstrum_segment_page(gray, binary, args):
    comps = get_connected_components(binary,
                                     min_area=args.cc_min_area,
                                     max_area=args.cc_max_area)
    if not comps:
        return []
    nn     = build_nn_graph(comps, k=args.k)
    ori    = estimate_page_orientation(nn)
    stroke = estimate_stroke_height(comps)
    spacing= estimate_char_spacing(nn, ori)
    clusters = build_docstrum_lines(
        comps, ori, spacing, stroke,
        min_dist_factor=args.min_dist_factor,
        max_dist_factor=args.max_dist_factor,
        angle_tol=args.angle_tol
    )
    lines = clusters_to_boxes(clusters, comps)
    paras = dynamic_morphological_merge(
        gray.shape, lines,
        stroke_height=stroke,
        dynamic_pct=args.dynamic_pct,
        open_factor=args.open_factor
    )
    return paras

def compute_features_for_boxes(boxes, gray, binary):
    H, W = gray.shape
    feats = []
    for x, y, w, h in boxes:
        x0, y0, w0, h0 = map(int, (x, y, w, h))
        area = max(w0 * h0, 1)
        roi_gray = gray[y0:y0+h0, x0:x0+w0]
        roi_bin  = binary[y0:y0+h0, x0:x0+w0]
        feats.append([
            (y0 + h0/2) / H,
            float(np.std(roi_gray)),
            float((roi_bin > 0).sum()) / area,
            float(np.mean(roi_gray)),
            (w0 / h0) if h0 > 0 else 0.0,
            float((cv2.Canny(roi_gray,100,200)>0).sum()) / area,
            area / (W * H),
            (x0 + w0/2) / W
        ])
    return np.array(feats)

def draw_boxes(img, boxes, color=(0,255,0), labels=None):
    out = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        x0, y0, w0, h0 = map(int, (x, y, w, h))
        cv2.rectangle(out, (x0, y0), (x0+w0, y0+h0), color, 2)
        if labels:
            cv2.putText(out, labels[i], (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out

def draw_boxes_with_preds(img, boxes, preds, palette, label_map):
    out = img.copy()
    for (x, y, w, h), cid in zip(boxes, preds):
        x0, y0, w0, h0 = map(int, (x, y, w, h))
        col = palette[cid % len(palette)]
        txt = label_map.get(cid, f"Class {cid}")
        cv2.rectangle(out, (x0, y0), (x0+w0, y0+h0), col, 2)
        cv2.putText(out, txt, (x0, y0-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return out

# ─── Streamlit UI ───────────────────────────────────────────────────
st.title("DocVision Inference Pipeline")

# 1) Upload
uploaded = st.file_uploader("Upload a document image",
                            type=["png","jpg","jpeg"])
if not uploaded:
    st.stop()

# 2) Preprocess
img_pil  = Image.open(uploaded).convert("RGB")
arr_rgb  = np.array(img_pil)
gray     = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
_, binary= cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3) Segmentation
xy_boxes     = recursive_xy_cut_debug(
    binary, x0=0, y0=0,
    min_block=seg_args.xy_min_block,
    thr_h=seg_args.xy_thr_h, thr_v=seg_args.xy_thr_v,
    gap_h=seg_args.xy_gap_h, gap_v=seg_args.xy_gap_v,
    k_h=seg_args.xy_k_h, k_v=seg_args.xy_k_v
)
doc_boxes    = docstrum_segment_page(gray, binary, seg_args)
hybrid_boxes = hybrid_segment_page(gray, binary, seg_args)

st.subheader("Segmentation Results")
cols = st.columns(3)
for col, title, boxes in zip(
    cols, ["XY‑Cut", "DocStrum", "Hybrid"],
    [xy_boxes, doc_boxes, hybrid_boxes]
):
    col.image(draw_boxes(arr_rgb, boxes),
              caption=title,
              use_container_width=True)

# 4) Classification
if st.button("Next: Classify zones"):
    try:
        clf = joblib.load(PKL_PATH)
    except ModuleNotFoundError:
        st.error("LightGBM missing—pip install lightgbm")
        st.stop()

    # find out how many classes the model was trained on
    ncls = clf.params.get("num_class", None)
    if ncls is None:
        st.error("Could not read num_class from model.")
        st.stop()

    # build a dynamic index→COCO-ID map  {0:1,1:2,...,10:11}
    idx2coco = {i: i+1 for i in range(ncls)}

    # predict
    feats = compute_features_for_boxes(xy_boxes, gray, binary)
    raw   = clf.predict(feats)
    arr   = np.asarray(raw)
    if arr.ndim == 2:
        idxs_xy = np.argmax(arr, axis=1).tolist()
    else:
        idxs_xy = [int(x) for x in arr.tolist()]
    xy_ids = [idx2coco[i] for i in idxs_xy]

    feats = compute_features_for_boxes(doc_boxes, gray, binary)
    raw   = clf.predict(feats)
    arr   = np.asarray(raw)
    if arr.ndim == 2:
        idxs_doc = np.argmax(arr, axis=1).tolist()
    else:
        idxs_doc = [int(x) for x in arr.tolist()]
    doc_ids = [idx2coco[i] for i in idxs_doc]

    feats = compute_features_for_boxes(hybrid_boxes, gray, binary)
    raw   = clf.predict(feats)
    arr   = np.asarray(raw)
    if arr.ndim == 2:
        idxs_hy = np.argmax(arr, axis=1).tolist()
    else:
        idxs_hy = [int(x) for x in arr.tolist()]
    hy_ids = [idx2coco[i] for i in idxs_hy]

    # overlay
    palette   = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    st.subheader("Classification Results")
    cols2 = st.columns(3)
    for col, title, boxes, cids in zip(
        cols2,
        ["XY‑Cut","DocStrum","Hybrid"],
        [xy_boxes, doc_boxes, hybrid_boxes],
        [xy_ids,   doc_ids,   hy_ids]
    ):
        col.image(
            draw_boxes_with_preds(arr_rgb, boxes, cids, palette, CATEGORY_MAP),
            caption=title,
            use_container_width=True
        )

    # save JSON
    results = []
    for method, boxes, cids in [
        ("xycut", xy_boxes, xy_ids),
        ("docstrum", doc_boxes, doc_ids),
        ("hybrid", hybrid_boxes, hy_ids)
    ]:
        for (x, y, w, h), cid in zip(boxes, cids):
            results.append({
                "method":        method,
                "bbox":          [int(x), int(y), int(w), int(h)],
                "category_id":   cid,
                "category_name": CATEGORY_MAP[cid]
            })

    out_path = "classification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    st.success(f"Saved results to `{out_path}`")
    st.download_button(
        "Download JSON",
        data=json.dumps(results, indent=2),
        file_name=out_path,
        mime="application/json"
    )

# 5) Optional Faster R‑CNN panel
with st.expander("🚀 Try Faster R‑CNN"):
    if st.checkbox("Run Faster R‑CNN"):
        try:
            num_cls = max(CATEGORY_MAP.keys()) + 1
            model = load_fastrcnn_model(
            repo_id="pmodi08/DocVision-Models",
            filename="fasterrcnn_doclaynet.pth",
            num_classes=max(CATEGORY_MAP.keys()) + 1
        )
        except FileNotFoundError:
            st.error(f"Checkpoint not found")
        else:
            thresh = st.slider("Score threshold", 0.0, 1.0, 0.5)
            tensor = F.to_tensor(arr_rgb)
            with torch.no_grad():
                out = model([tensor])[0]
            b, l, s = (out["boxes"].cpu().numpy(),
                       out["labels"].cpu().numpy(),
                       out["scores"].cpu().numpy())
            mask    = s >= thresh
            b, l    = b[mask], l[mask]
            st.image(
                draw_boxes_with_preds(arr_rgb, b, l,
                                      [(255,0,0),(0,255,0),(0,0,255)],
                                      CATEGORY_MAP),
                caption="Faster R‑CNN Results",
                use_container_width=True
            )
