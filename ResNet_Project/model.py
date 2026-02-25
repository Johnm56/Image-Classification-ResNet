import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, num_classes=20):
        super(BaseModel, self).__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the final FC layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )

        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4, and fc
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in ["layer4", "fc"]):
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
        