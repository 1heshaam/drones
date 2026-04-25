from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


MODEL_PATH = Path("normal_sky_model.pt")
CLASS_NAMES_PATH = Path("class_names.json")
FEATURE_INDEX_PATH = Path("feature_index.pt")

TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/val")

BATCH_SIZE = 16


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FeatureExtractor(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.features = nn.Sequential(*list(trained_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def load_model(class_names, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    return model


def extract_features(feature_model, loader, device):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = feature_model(images)

            features = torch.nn.functional.normalize(features, dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features), torch.cat(all_labels)


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(class_names, device)
    feature_model = FeatureExtractor(model)
    feature_model.eval()
    feature_model.to(device)

    train_features, train_labels = extract_features(feature_model, train_loader, device)
    val_features, val_labels = extract_features(feature_model, val_loader, device)

    centroids = {}
    thresholds = {}

    for class_idx, class_name in enumerate(class_names):
        class_train_features = train_features[train_labels == class_idx]

        centroid = class_train_features.mean(dim=0)
        centroid = torch.nn.functional.normalize(centroid, dim=0)

        centroids[class_name] = centroid

        class_val_features = val_features[val_labels == class_idx]

        if len(class_val_features) > 0:
            similarities = class_val_features @ centroid
            distances = 1 - similarities

            # Conservative threshold: allow most validation examples, but reject far-away samples
            threshold = torch.quantile(distances, 0.90).item()
        else:
            threshold = 0.35

        thresholds[class_name] = threshold

        print(f"{class_name}: threshold={threshold:.4f}")

    torch.save(
        {
            "class_names": class_names,
            "centroids": centroids,
            "thresholds": thresholds,
        },
        FEATURE_INDEX_PATH
    )

    print(f"Saved feature index to {FEATURE_INDEX_PATH}")


if __name__ == "__main__":
    main()