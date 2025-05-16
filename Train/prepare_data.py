import os
import pandas as pd
from feature_extraction import save_features

def prepare_data(
    train_json: str,
    val_json:   str,
    image_dir:  str,
    feature_dir: str = '.'
):
    # Categories to exclude
    EXCLUDE = {'Caption', 'Footnote', 'List-item'}

    # Paths to feature files
    train_csv = os.path.join(feature_dir, 'updated_features_train.csv')
    val_csv   = os.path.join(feature_dir, 'updated_features_val.csv')

    # Decide whether to reuse existing features
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        resp = input(
            f"Found existing feature files:\n"
            f"  {train_csv}\n"
            f"  {val_csv}\n"
            "Load these instead of re-extracting? [Y/n]: "
        ).strip().lower()
        use_existing = (resp == '' or resp.startswith('y'))
    else:
        use_existing = False

    # Load or extract train features
    if use_existing:
        train_feat = pd.read_csv(train_csv)
    else:
        train_feat = save_features(train_json, image_dir, feature_dir)

    # Load or extract val features
    if use_existing:
        val_feat = pd.read_csv(val_csv)
    else:
        val_feat = save_features(val_json, image_dir, feature_dir)

    # --- Filter out unwanted categories ---
    train_feat = train_feat[~train_feat['category'].isin(EXCLUDE)]
    val_feat   = val_feat[~val_feat['category'].isin(EXCLUDE)]

    # --- Select only the top‐important features & labels ---
    FEATURES_TO_USE = [
        'y_center_norm',
        'std_intensity',
        'text_density',
        'mean_intensity',
        'aspect_ratio',
        'edge_density',
        'area_norm',
        'x_center_norm'
    ]

    missing = set(FEATURES_TO_USE) - set(train_feat.columns)
    if missing:
        raise ValueError(f"Missing features in train_feat: {missing}")

    X_train = train_feat[FEATURES_TO_USE].copy()
    y_train = train_feat['category']

    X_val   = val_feat[FEATURES_TO_USE].copy()
    y_val   = val_feat['category']

    # --- Use raw COCO IDs (shifted to 0…K–1) ---
    y_train_id = train_feat['category_id'].astype(int) - 1
    y_val_id   = val_feat  ['category_id'].astype(int) - 1

    return X_train, X_val, y_train_id, y_val_id

if __name__ == "__main__":
    base      = '../Dataset/DocLayNet/DocLayNet_core'
    coco      = os.path.join(base, 'COCO')
    png       = os.path.join(base, 'PNG')
    feature_d = '.'

    train_json = os.path.join(coco, 'train.json')
    val_json   = os.path.join(coco, 'val.json')

    X_tr, X_va, y_tr, y_va, le = prepare_data(
        train_json,
        val_json,
        png,
        feature_d
    )
    print(f"\nTrain samples: {len(X_tr)}, Val samples: {len(X_va)}")
    print("Classes:", list(le.classes_))
