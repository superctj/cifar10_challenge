"""
Wide-Resnet-34 Classifier for 32x32x3 CIFAR-10 Image Classification.
Reference: https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/cifar.py
"""
import numpy as np
import torch
import torch.nn as nn
from armory.data.utils import maybe_download_weights_from_s3
from art.classifiers import PyTorchClassifier

from arch.model import make_wideresnet_model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocessing_fn(img):
    img = img.astype(np.float32) / 255.0
    img = img.transpose(0, 3, 1, 2)  # from NHWC to NCHW

    return img


def get_wideresnet(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_wideresnet_model(**model_kwargs)
    model.to(DEVICE)

    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(),
                                  lr = 0.1,
                                  momentum=0.9,
                                  weight_decay=2e-4),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs
    )

    return wrapped_model