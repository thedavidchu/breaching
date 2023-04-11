"""Helper code to instantiate various models."""

import torch
import torchvision

from collections import OrderedDict

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG

from .language_models import RNNModel, TransformerModel, LinearModel
from .losses import CausalLoss, MLMLoss, MostlyCausalLoss


def construct_model(pretrained=True):
    # cfg_data.modality == "vision":
    model = _construct_vision_model(pretrained)
    # Save nametag for printouts later:
    model.name = "ResNet18"

    # Choose loss function according to data and model:
    # "classification" in cfg_data.task:
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.jit.script(loss_fn)
    return model, loss_fn


class VisionContainer(torch.nn.Module):    # NOTE(dchu): FISHING
    """We'll use a container to catch extra attributes and allow for usage with model(**data)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, **kwargs):
        return self.model(inputs)


def _construct_vision_model(pretrained=True):
    """Construct the neural net that is used."""
    classes = 397   # cfg_data.classes
    model = torchvision.models.resnet18(pretrained=pretrained)
    fc = torch.nn.Linear(model.fc.in_features, classes)
    if pretrained:
        fc.weight.data = model.fc.weight[:classes]
        fc.bias.data = model.fc.bias[:classes]
    model.fc = fc

    return VisionContainer(model)
