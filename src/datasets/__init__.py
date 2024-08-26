from . import mnist, synthetic_aes

_DATASET_MODULES = {
    'mnist': mnist,
    'synthetic-aes': synthetic_aes.module
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