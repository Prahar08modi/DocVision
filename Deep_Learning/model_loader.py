# model_loader.py

from huggingface_hub import hf_hub_download
import joblib
import torch
import torchvision


def load_fastrcnn_model(
    repo_id: str,
    filename: str,
    num_classes: int
):
    """
    Download, build, and load the Faster R‑CNN model.
    """
    # download checkpoint
    path = hf_hub_download(repo_id=repo_id, filename=filename)

    # instantiate architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False,
        num_classes=num_classes
    )

    # load and eval
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    num_cls = 12  # adjust to your use‑case
    rcnn = load_fastrcnn_model(
        repo_id="pmodi08/DocVision-Models",
        filename="fasterrcnn_doclaynet.pth",
        num_classes=num_cls
    )
    print("Loaded Faster R‑CNN, modules:", len(list(rcnn.parameters())))
