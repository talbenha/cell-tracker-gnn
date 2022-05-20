"""This code was imported from:
    https://github.com/tamerthamoqa/facenet-pytorch-glint360k
"""

import torch.nn as nn
from torch.nn import functional as F
from .utils_resnet import resnet18, resnet34
from pytorch_metric_learning.utils import common_functions

class Resnet18Triplet(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.
    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, pretrained=False):
        super(Resnet18Triplet, self).__init__()
        self.model = resnet18(pretrained=pretrained)

        # Output
        self.input_features_fc_layer = self.model.fc.in_features
        self.model.fc = common_functions.Identity()

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)

        return embedding


class Resnet34Triplet(nn.Module):
    """Constructs a ResNet-34 model for FaceNet training using triplet loss.
    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, pretrained=False):
        super(Resnet34Triplet, self).__init__()
        self.model = resnet34(pretrained=pretrained)

        # Output
        self.input_features_fc_layer = self.model.fc.in_features
        self.model.fc = common_functions.Identity()

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)

        return embedding


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False, normalized_feat=False):
        super().__init__()
        self.normalized_feat = normalized_feat
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        x = self.net(x)
        if self.normalized_feat:
            x = F.normalize(x, p=2, dim=1) # from torch.nn import functional as F
        return x

def set_model_architecture(model_architecture):
    if model_architecture == "resnet18":
        model = Resnet18Triplet()
    elif model_architecture == "resnet34":
        model = Resnet34Triplet()
    print("Using {} model architecture.".format(model_architecture))

    return model

