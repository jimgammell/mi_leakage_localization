from .base_model import GenericZaidNet, GenericWoutersNet
from .keras_to_pytorch_utils import *

class ZaidNet__ASCADv1f(GenericZaidNet):
    def __init__(self, pretrained_param_dir=None):
        super().__init__(
            block_settings=[{'channels': 4, 'conv_kernel_size': 1, 'pool_kernel_size': 2}],
            dense_widths=[10, 10]
        )
        self.pretrained_param_dir = pretrained_param_dir
        if self.pretrained_param_dir is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_param_dir is not None
        keras_params = unpack_keras_params(self.pretrained_param_dir)
        keras_to_torch_mod(keras_params['block1_conv1'], self.conv_stage.block_1.conv)
        keras_to_torch_mod(keras_params['block1_norm1'], self.conv_stage.block_1.norm)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['fc2'], self.fc_stage.dense_2)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class ZaidNet__DPAv4(GenericZaidNet):
    def __init__(self, pretrained_param_dir=None):
        super().__init__(
            block_settings=[{'channels': 2, 'conv_kernel_size': 1, 'pool_kernel_size': 2}],
            dense_widths=[2]
        )
        self.pretrained_param_dir = pretrained_param_dir
        if self.pretrained_param_dir is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_param_dir is not None
        keras_params = unpack_keras_params(self.pretrained_param_dir)
        keras_to_torch_mod(keras_params['block1_conv1'], self.conv_stage.block_1.conv)
        keras_to_torch_mod(keras_params['block1_norm1'], self.conv_stage.block_1.norm)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class ZaidNet__AES_HD(GenericZaidNet):
    def __init__(self, pretrained_param_dir=None):
        super().__init__(
            block_settings=[{'channels': 2, 'conv_kernel_size': 1, 'pool_kernel_size': 2}],
            dense_widths=[2]
        )
        self.pretrained_param_dir = pretrained_param_dir
        if self.pretrained_param_dir is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_param_dir is not None
        keras_params = unpack_keras_params(self.pretrained_param_dir)
        keras_to_torch_mod(keras_params['block1_conv1'], self.conv_stage.block_1.conv)
        keras_to_torch_mod(keras_params['block1_norm1'], self.conv_stage.block_1.norm)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class WoutersNet__ASCADv1f(GenericZaidNet):
    def __init__(self, pretrained_param_dir=None):
        super().__init__(
            input_pool_size=2,
            dense_widths=[10, 10]
        )
        self.pretrained_param_dir = pretrained_param_dir
        if self.pretrained_param_dir is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_param_dir is not None
        keras_params = unpack_keras_params(self.pretrained_param_dir)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['fc2'], self.fc_stage.dense_2)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class WoutersNet__DPAv4(GenericZaidNet):
    def __init__(self, pretrained_param_dir=None):
        super().__init__(
            input_pool_size=2,
            dense_widths=[2]
        )
        self.pretrained_param_dir = pretrained_param_dir
        if self.pretrained_param_dir is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_param_dir is not None
        keras_params = unpack_keras_params(self.pretrained_param_dir)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class WoutersNet__AES_HD(GenericZaidNet):
    def __init__(self, pretrained_param_dir=None):
        super().__init__(
            input_pool_size=2,
            dense_widths=[2]
        )
        self.pretrained_param_dir = pretrained_param_dir
        if self.pretrained_param_dir is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_param_dir is not None
        keras_params = unpack_keras_params(self.pretrained_param_dir)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)