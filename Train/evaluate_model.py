
import os
import joblib
import numpy as np
import pandas as pd
from load_coco_annotations import load_coco_annotations
from feature_extraction import save_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
BASE_DIR    = '../Dataset/DocLayNet/DocLayNet_core'
COCO_DIR    = os.path.join(BASE_DIR, 'COCO')
PNG_DIR     = os.path.join(BASE_DIR, 'PNG')
FEAT_DIR    = '../Feature_Extraction'
MODEL_PATH  = '../Classification/lightgbm_doclaynet.pkl'
OUT_CSV     = 'predictions_test.csv'

# Categories to exclude by COCO ID
EXCLUDE_IDS = {1, 2, 4}  # Caption=1, Footnote=2, List-item=4

# 1. Load model
model = joblib.load(MODEL_PATH)
print(f"▶️  Loaded model from {MODEL_PATH}")

# 2. Load or extract test features
feat_csv = os.path.join(FEAT_DIR, 'updated_features_test.csv')
if not os.path.exists(feat_csv):
    print("[→] No features found; extracting now...")
    save_features(os.path.join(COCO_DIR, 'test.json'), PNG_DIR, FEAT_DIR)
df = pd.read_csv(feat_csv)

# 3. Load annotation mapping for category names
ann_df = load_coco_annotations(os.path.join(COCO_DIR, 'test.json'))
cat_map = dict(zip(ann_df['category_id'], ann_df['category']))

# 4. Filter out unwanted categories
df = df[~df['category_id'].isin(EXCLUDE_IDS)]  # preserve original index
print(f"▶️  Evaluating on {len(df)} samples after filtering IDs {EXCLUDE_IDS}")

# 5. Select feature columns (must match training)
FEATURES = [
    'y_center_norm',
    'std_intensity',
    'text_density',
    'mean_intensity',
    'aspect_ratio',
    'edge_density',
    'area_norm',
    'x_center_norm'
]
missing = set(FEATURES) - set(df.columns)
if missing:
    raise ValueError(f"Missing features in test set: {missing}")
X_test = df[FEATURES]

# 6. True labels (0…K-1)
y_true = df['category_id'].astype(int) - 1

# 7. Predict probabilities and classes
y_prob = model.predict(X_test)
# handle both Booster.predict and sklearn.predict_proba
if y_prob.ndim > 1:
    y_pred0 = np.argmax(y_prob, axis=1)
else:
    y_pred0 = y_prob.astype(int)
# convert back to original COCO IDs
y_pred_id = (y_pred0 + 1)

# 8. Append predictions to DataFrame
df['predicted_category_id'] = y_pred_id
df['predicted_category']    = df['predicted_category_id'].map(cat_map)

# 9. Compute metrics on numeric labels
y_pred = y_pred0
n_true = y_true
acc = accuracy_score(n_true, y_pred)
print(f"\n▶️  Test Accuracy: {acc:.4f}\n")
print("▶️  Classification Report:")
# Map back to names for report
def id2name(id0): return cat_map[id0+1]
target_names = [id2name(i) for i in sorted(set(n_true))]
print(classification_report(n_true, y_pred, target_names=target_names))

# Confusion matrix
print("▶️  Confusion Matrix:")
cm = confusion_matrix(n_true, y_pred)
row_names = [f"true_{n}" for n in sorted(set(n_true+1))]
col_names = [f"pred_{n}" for n in sorted(set(y_pred_id))]
cm_df = pd.DataFrame(cm, index=row_names, columns=col_names)
print(cm_df)

# 10. Save predictions
df.to_csv(OUT_CSV, index=False)
print(f"✅  Saved predictions with labels to {OUT_CSV}")

