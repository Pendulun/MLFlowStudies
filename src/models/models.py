from __future__ import annotations
from abc import ABC, abstractmethod

from collections import namedtuple

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import googlenet, GoogLeNet_Weights

ModelAttr = namedtuple('ModelAttr', ['weights', 'preprocess', 'model'])

class ModelCreator(ABC):

    @abstractmethod
    def factory_method(self, device:str):
        pass

class ResnetModelCreator(ABC):

    def factory_method(self, device:str) -> Model:
        return ResnetModel(device)

class GoogLeNetModelCreator(ABC):

    def factory_method(self, device:str) -> Model:
        return GoogLeNetModel(device)

class Model(ABC):
    def __init__(self, device:str):
        self._device = self._define_real_device(device)
        self._model_attr = self._define_model_attr()

        self._model_attr.model.to(torch.device(device))

    def _define_real_device(self, device) -> str:
        if device[:4] == "cuda":
            device = device if self._is_cuda_available() else "cpu"
        
        return device
    
    def _is_cuda_available(self):
        return torch.cuda.is_available()
    
    @abstractmethod
    def _define_model_attr(self):
        pass
    
    @property
    def model(self):
        return self._model_attr.model
    
    @model.setter
    def model(self, model):
        self._model_attr.model = model
    
    @property
    def weights(self):
        return self._model_attr.weights

    @weights.setter
    def weights(self, weights):
        raise AttributeError("weights is read only")
    
    @property
    def preprocess(self):
        return self._model_attr.preprocess
    
    @preprocess.setter
    def preprocess(self, preprocess):
        self._model_attr.preprocess = preprocess

class ResnetModel(Model):

    def __init__(self, device:str):
        super().__init__(device)
    
    def _define_model_attr(self):
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        model = resnet50(weights=weights)

        attrbs = ModelAttr(weights, preprocess, model)
        return attrbs

class GoogLeNetModel(Model):
    def __init__(self, device:str):
        super().__init__(device)

    def _define_model_attr(self):
        weights = GoogLeNet_Weights.DEFAULT
        preprocess = weights.transforms()

        model = googlenet(weights=weights)

        attrbs = ModelAttr(weights, preprocess, model)
        return attrbs