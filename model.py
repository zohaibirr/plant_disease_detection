import torch.nn as nn
import timm


class SimpleCNN(nn.Module):
    """
    A small CNN baseline for plant disease classification.
    """
    def __init__(self, num_classes: int, image_size: int = 128):
        super().__init__()
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        feat_size = image_size // 8  # after 3 maxpools
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_size * feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_vit(
    num_classes: int,
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
