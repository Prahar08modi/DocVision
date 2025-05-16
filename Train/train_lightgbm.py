import argparse
import os
import numpy as np
import joblib
import lightgbm as lgb
from prepare_data import prepare_data

def parse_args():
    p = argparse.ArgumentParser(
        description="Train LightGBM on DocLayNet features (GPU or CPU)"
    )
    p.add_argument(
        "--device", choices=["gpu", "cpu"], default="gpu",
        help="Compute device to use for training"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1. Load features & labels
    X_train, X_val, y_train, y_val = prepare_data(
        '../Dataset/DocLayNet/DocLayNet_core/COCO/train.json',
        '../Dataset/DocLayNet/DocLayNet_core/COCO/val.json',
        '../Dataset/DocLayNet/DocLayNet_core/PNG'
    )

    # 2. Create LightGBM dataset for CV
    lgb_train = lgb.Dataset(X_train, label=y_train)

    # 3. Base parameters
    params = {
        'objective': 'multiclass',
        'num_class': int(y_train.max() + 1),
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'verbose': -1
    }

    # 4. Device selection
    if args.device == "gpu":
        params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        })
        print("🖥️  Training with GPU acceleration")
    else:
        # ensure no GPU params remain
        for k in ['device', 'gpu_platform_id', 'gpu_device_id']:
            params.pop(k, None)
        print("⚙️  Training on CPU")

    # 5. Cross-validation with early stopping via callbacks
    print("🔄  Running LightGBM CV with early stopping...")
    cv_results = lgb.cv(
        params,
        lgb_train,
        nfold=3,
        num_boost_round=2000,
        stratified=True,
        seed=42,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    # 6. Figure out which key holds the mean metric
    metric = 'multi_logloss'
    mean_key = next(
        (k for k in cv_results.keys() if metric in k and 'mean' in k),
        None
    )
    if mean_key is None:
        raise KeyError(
            f"Could not find a key containing '{metric}' and 'mean'. "
            f"Available keys: {list(cv_results.keys())}"
        )

    best_rounds = len(cv_results[mean_key])
    print(f"✅  Best num_boost_round from CV: {best_rounds}")

    # 7. Train final model on train+val
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    lgb_full = lgb.Dataset(X_full, label=y_full)

    print("🚀  Training final model on train+val...")
    final_model = lgb.train(
        params,
        lgb_full,
        num_boost_round=best_rounds
    )

    # 8. Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/updated_lightgbm_doclaynet.pkl')

    print("💾  Saved final model to models/updated_lightgbm_doclaynet.pkl")

if __name__ == "__main__":
    main()
