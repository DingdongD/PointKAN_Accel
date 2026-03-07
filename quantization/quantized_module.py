# Ported from ISEDA26/iseda_quant for PointKAN-Elite (RTN: Linear/Conv1d/Conv2d + MinMax + FixedFakeQuantize)
from __future__ import absolute_import
from torch import nn
from .observer import MinMaxObserver
from .fake_quant import FixedFakeQuantize

ObserverDict = {"MinMaxObserver": MinMaxObserver}
FakeQuantizeDict = {"FixedFakeQuantize": FixedFakeQuantize}


def ActivationQuantizer(a_qconfig):
    return FakeQuantizeDict[a_qconfig.quantizer](
        ObserverDict[a_qconfig.observer],
        bit=a_qconfig.bit,
        symmetric=a_qconfig.symmetric,
        ch_axis=a_qconfig.ch_axis,
    )


def WeightQuantizer(w_qconfig):
    return FakeQuantizeDict[w_qconfig.quantizer](
        ObserverDict[w_qconfig.observer],
        bit=w_qconfig.bit,
        symmetric=w_qconfig.symmetric,
        ch_axis=w_qconfig.ch_axis,
    )


def get_module_args(module):
    if isinstance(module, nn.Linear):
        return dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
    if isinstance(module, nn.Conv1d):
        return dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
        )
    if isinstance(module, nn.Conv2d):
        return dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
    raise NotImplementedError(f"Module type {type(module)} not supported.")


module_type_to_quant_weight = {nn.Linear: None, nn.Conv1d: None, nn.Conv2d: None}


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias, w_qconfig):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input, gamma=None):
        return nn.functional.linear(
            input, self.weight_fake_quant(self.weight), self.bias
        )


class QConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, bias, w_qconfig):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input, gamma=None):
        return self._conv_forward(
            input, self.weight_fake_quant(self.weight), self.bias
        )


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, bias, padding_mode, w_qconfig):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode,
        )
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input, gamma=None):
        return self._conv_forward(
            input, self.weight_fake_quant(self.weight), self.bias
        )


module_type_to_quant_weight[nn.Linear] = QLinear
module_type_to_quant_weight[nn.Conv1d] = QConv1d
module_type_to_quant_weight[nn.Conv2d] = QConv2d


def Quantizer(module, config, sign=False):
    if module is None:
        return ActivationQuantizer(a_qconfig=config)
    module_type = type(module)
    original_device = next(module.parameters()).device
    if module_type in module_type_to_quant_weight:
        kwargs = get_module_args(module)
        qmodule = module_type_to_quant_weight[module_type](**kwargs, w_qconfig=config)
        qmodule.weight.data = module.weight.data.clone()
        if getattr(module, "bias", None) is not None:
            qmodule.bias.data = module.bias.data.clone()
        return qmodule.to(original_device)
    return module


class QuantizedModule(nn.Module):
    def __init__(self):
        super().__init__()


class QuantizedLayer(QuantizedModule):
    def __init__(self, module, activation, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        original_device = next(module.parameters()).device
        if qoutput:
            self.layer_post_act_fake_quantize = Quantizer(None, a_qconfig).to(original_device)
        self.module = Quantizer(module, w_qconfig).to(original_device)
        self.qoutput = qoutput
        self.activation = activation

    def forward(self, x):
        x = self.module(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.qoutput:
            x = self.layer_post_act_fake_quantize(x)
        return x


class PreQuantizedLayer(QuantizedModule):
    def __init__(self, module, activation, w_qconfig, a_qconfig, qinput=True):
        super().__init__()
        original_device = next(module.parameters()).device
        self.qinput = qinput
        if qinput:
            self.act_fake_quant = Quantizer(None, a_qconfig).to(original_device)
        self.module = Quantizer(module, w_qconfig).to(original_device)
        self.activation = activation

    def forward(self, x, gamma=None):
        if self.qinput:
            x = self.act_fake_quant(x)
        x = self.module(x, gamma)
        if self.activation is not None:
            x = self.activation(x)
        return x
