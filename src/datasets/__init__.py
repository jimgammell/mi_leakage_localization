from . import mnist, synthetic_aes, simple_gaussian

_DATASET_MODULES = {
    'mnist': mnist,
    'synthetic-aes': synthetic_aes.module,
    'simple-gaussian': simple_gaussian
}
AVAILABLE_DATASETS = list(_DATASET_MODULES.keys())

def _check_name(name):
    if not name in AVAILABLE_DATASETS:
        raise NotImplementedError(f'Unrecognized dataset name: {name}.')

def download(name, **kwargs):
    _check_name(name)
    _DATASET_MODULES[name].download(**kwargs)

def load(name, **kwargs):
    _check_name(name)
    dataset_module = _DATASET_MODULES[name].DataModule(**kwargs)
    return dataset_module