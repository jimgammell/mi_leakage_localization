from .multilayer_perceptron import MultilayerPerceptron
from .lenet import LeNet5
from .sca_cnn import SCA_CNN

_MODEL_CONSTRUCTORS = {
    'multilayer-perceptron': MultilayerPerceptron,
    'lenet-5': LeNet5,
    'sca-cnn': SCA_CNN
}
AVAILABLE_MODELS = list(_MODEL_CONSTRUCTORS.keys())

def load(name, **kwargs):
    if not(name in AVAILABLE_MODELS):
        raise NotImplementedError(f'Unrecognized model name: {name}.')
    model = _MODEL_CONSTRUCTORS[name](**kwargs)
    return model