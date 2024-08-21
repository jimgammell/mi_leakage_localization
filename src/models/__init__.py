from utils.constant import Constant
from .lenet import LeNet5

_MODEL_CONSTRUCTORS = Constant({
    'lenet-5': LeNet5
})
AVAILABLE_MODELS = Constant(list(_MODEL_CONSTRUCTORS.keys()))

def load(name, **kwargs):
    if not(name in AVAILABLE_MODELS):
        raise NotImplementedError(f'Unrecognized model name: {name}.')
    model = _MODEL_CONSTRUCTORS[name](**kwargs)
    return model