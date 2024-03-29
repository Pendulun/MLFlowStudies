import unittest
from models import ModelCreator
from models import Model

from models import ResnetModel
from models import ResnetModelCreator

from models import GoogLeNetModel
from models import GoogLeNetModelCreator

class ModelCreatorTest(unittest.TestCase):
    
    def get_resnet_model(self, device:str = 'cpu') -> ResnetModel:
        creator = ResnetModelCreator()
        return creator.factory_method(device)
    
    def get_googlenet_model(self, device:str = 'cpu') -> GoogLeNetModel:
        creator = GoogLeNetModelCreator()
        return creator.factory_method(device)

    def test_cant_instantiate_abstract_creator(self):
        with self.assertRaises(TypeError):
            creator = ModelCreator()
    
    def test_cant_instantiate_abstract_model(self):
        with self.assertRaises(TypeError):
            model = Model()

    def test_can_create_resnet_model(self):
        model = self.get_resnet_model()
        self.assertIsInstance(model, ResnetModel)
    
    def test_can_create_googlenet_model(self):
        model = self.get_googlenet_model()
        self.assertIsInstance(model, GoogLeNetModel)

class ModelsTeste(unittest.TestCase):
    def get_resnet_model(self, device:str = 'cpu') -> ResnetModel:
        creator = ResnetModelCreator()
        return creator.factory_method(device)
    
    def test_can_get_model_cpu_device(self):
        target_device = 'cpu'
        model = self.get_resnet_model(device=target_device)
        self.assertEqual(model.device, target_device)
    
    def test_cant_override_model_device(self):
        target_device = 'cpu'
        model = self.get_resnet_model(device=target_device)

        with self.assertRaises(AttributeError):
            new_device = 'cuda'
            model.device = new_device
    
    def test_get_model_preprocess(self):
        model = self.get_resnet_model()
        self.assertIsNotNone(model.preprocess)
    
    def test_can_set_model_preprocess(self):
        model = self.get_resnet_model()
        target_preprocess = "my_preprocess" 
        model.preprocess = target_preprocess
        self.assertEqual(model.preprocess, target_preprocess)
    
    def test_can_get_model_weights(self):
        model = self.get_resnet_model()
        self.assertIsNotNone(model.weights)
    
    def test_cant_override_model_weights(self):
        model = self.get_resnet_model()
        with self.assertRaises(AttributeError):
            model.weights = [1]


class ResnetModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = self.get_resnet_model()
    
    def get_resnet_model(self, device:str = 'cpu') -> ResnetModel:
        creator = ResnetModelCreator()
        return creator.factory_method(device)

    def test_can_get_resnet_model(self):
        self.assertIsNotNone(self._model.model)
    
    def test_get_resnet_model_weights(self):
        self.assertIsNotNone(self._model.weights)
    
class GoogLeNetModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = self.get_googlenet_model()
    
    def get_googlenet_model(self, device:str = 'cpu') -> GoogLeNetModel:
        creator = GoogLeNetModelCreator()
        return creator.factory_method(device)

    def test_can_get_googlenet_model(self):
        self.assertIsNotNone(self._model.model)
    
    def test_get_googlenet_model_weights(self):
        self.assertIsNotNone(self._model.weights)
    
if __name__ == "__main__":
    unittest.main()