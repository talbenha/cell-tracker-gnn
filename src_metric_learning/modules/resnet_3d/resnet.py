"""This code was imported from:
    https://github.com/tamerthamoqa/facenet-pytorch-glint360k
"""

import torch.nn as nn
from torch.nn import functional as F
from .utils_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from pytorch_metric_learning.utils import common_functions


class Resnet18_3D(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.
    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.

    """

    def __init__(self, embedding_dimension=512):
        super(Resnet18_3D, self).__init__()
        self.model = resnet18()

        # Output embedding
        self.input_features_fc_layer = self.model.fc.in_features
        self.model.fc = common_functions.Identity() #nn.Linear(input_features_fc_layer, embedding_dimension, bias=False)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        # embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class Resnet34_3D(nn.Module):
    """Constructs a ResNet-34 model for FaceNet training using triplet loss.
    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.

    """

    def __init__(self, embedding_dimension=512):
        super(Resnet34_3D, self).__init__()
        self.model = resnet34()

        # Output embedding
        self.input_features_fc_layer = self.model.fc.in_features
        self.model.fc = common_functions.Identity() # nn.Linear(input_features_fc_layer, embedding_dimension, bias=False)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        # embedding = F.normalize(embedding, p=2, dim=1)

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

def set_model_architecture(model_architecture): #, embedding_dimension):
    if model_architecture == "resnet18_3d":
        model = Resnet18_3D()
    elif model_architecture == "resnet34_3d":
        model = Resnet34_3D()
    print("Using {} model architecture.".format(model_architecture))

    return model

