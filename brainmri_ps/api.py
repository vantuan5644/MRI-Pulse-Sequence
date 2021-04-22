import os
import sys
from pathlib import Path
from typing import Union

module_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(module_path)

import torch
import torch.nn as nn
import albumentations as A
import albumentations.pytorch as AT
import gdown

from data import BrainMRIDataset
from models import create_model
from utils import dcm2image, study2series_dict

__all__ = ['PulseSequenceClassifier']

model_urls = {
    'mobilenet_v2': '1P7mR3RkGEIqbANmzqVXvU1Kwr4qXKgeq', }


class PulseSequenceClassifier():
    def __init__(self, model_name="mobilenet_v2", device="cpu"):
        """
        MRI Pulse Sequence classification object.

        The implementation is built from *torchvision*.

        For using pretrained weights, initialize the classifier object by using:
        `PulseSequenceClassifier().from_pretrained()`

        The classifier supports both instance level (with `predict_instance` method)
        and study level (with `predict_study` method).

        :param model_name: str, default="mobilenet_v2"

            Load model structure from `torchvision`, default="mobilenet_v2".
            Currently supported models are: "mobilenet_v2".
        :param device: str, default="cpu"
            Move model and data to a specific device, default="cpu".
            Change `device` to `cuda` in case of GPU inference.
        """
        self.device = torch.device(device)
        self.model_name = model_name

        supported_models = list(model_urls.keys())
        if model_name not in supported_models:
            raise NotImplementedError('Currently supported models are', list(model_urls.keys()))

        self.label_dict = dict([
            (0, "FLAIR"),
            (1, "T1C"),
            (2, "T2"),
            (3, "ADC"),
            (4, "DWI"),
            (5, "TOF"),
            (6, "OTHER"),
        ])

        self.model = create_model(self.model_name, pretrained=False, n_classes=len(self.label_dict.keys()))
        self.model = self.model.to(self.device)
        # self.model = nn.DataParallel(self.model)
        self.model = WrappedModel(self.model)

        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            AT.ToTensor()
        ])

    def from_pretrained(self):
        checkpoint_path = module_path + "/{}.pt".format(self.model_name)
        if not os.path.exists(checkpoint_path):
            checkpoint_url = f'https://drive.google.com/uc?id={model_urls[self.model_name]}'
            gdown.cached_download(checkpoint_url, checkpoint_path)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return self

    def predict_instance(self, instance_path: Union[Path, str]) -> str:
        """
        Predict the MRI pulse sequence of a single instance of a series
        :param instance_path: {Path, str}
            Path to the dicom instance
        :return: Pulse Sequence prediction string

        """
        image = dcm2image(instance_path)
        image = self.transform(image=image)["image"]

        with torch.no_grad():
            self.model.eval()
            image = image.unsqueeze(0).to(self.device)

            output = self.model(image)
            pred = torch.max(output, 1)[1].detach().cpu().numpy()

        return self.label_dict[pred[0]]

    def predict_study(self, study_path: Union[Path, str]):
        """
        Predict the MRI pulse sequence for all axial series in a study

        Returns:
        :param study_path: {Path, str}
            Path to the study directory (contains instance dicom files)
        :return: Pulse Sequence predictions
            {"Series Instance UID": Pulse_sequence_prediction}

        """
        self.series_dict = study2series_dict(study_path)
        self.loader = torch.utils.data.DataLoader(
            BrainMRIDataset(
                self.series_dict,
                image_transform=self.transform
            ),
            batch_size=len(self.series_dict), shuffle=False
        )

        with torch.no_grad():
            self.model.eval()
            series_uids, images = next(iter(self.loader))
            images = images.to(self.device)

            outputs = self.model(images)
            preds = torch.max(outputs, 1)[1].detach().cpu().numpy()

        for i, series_uid in enumerate(series_uids):
            self.series_dict[series_uid] = self.label_dict[preds[i]]

        return self.series_dict


class WrappedModel(nn.Module):
    """
    To load DataParallel checkpoint without calling DP module
    """
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)
