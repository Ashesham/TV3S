# ----------------------------------------------------------------------------
# Portions of this file have been modified by Ash in 2025.
# Original source: OpenMMLab/mmcv (Apache License 2.0).
# Modifications were made to support the working environment of TV3S.
# ----------------------------------------------------------------------------

import pytest
import torch
import torch.nn as nn

from mmcv.cnn.bricks import DepthwiseSeparableConvModule


def test_depthwise_separable_conv():
    with pytest.raises(AssertionError):
        # conv_cfg must be a dict or None
        DepthwiseSeparableConvModule(4, 8, 2, groups=2)

    # test default config
    conv = DepthwiseSeparableConvModule(3, 8, 2)
    assert conv.depthwise_conv.conv.groups == 3
    assert conv.pointwise_conv.conv.kernel_size == (1, 1)
    assert not conv.depthwise_conv.with_norm
    assert not conv.pointwise_conv.with_norm
    assert conv.depthwise_conv.activate.__class__.__name__ == 'ReLU'
    assert conv.pointwise_conv.activate.__class__.__name__ == 'ReLU'
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # test dw_norm_cfg
    conv = DepthwiseSeparableConvModule(3, 8, 2, dw_norm_cfg=dict(type='SyncBN', requires_grad=True))
    assert conv.depthwise_conv.norm_name == 'bn'
    assert not conv.pointwise_conv.with_norm
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # test pw_norm_cfg
    conv = DepthwiseSeparableConvModule(3, 8, 2, pw_norm_cfg=dict(type='SyncBN', requires_grad=True))
    assert not conv.depthwise_conv.with_norm
    assert conv.pointwise_conv.norm_name == 'bn'
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # test norm_cfg
    conv = DepthwiseSeparableConvModule(3, 8, 2, norm_cfg=dict(type='SyncBN', requires_grad=True))
    assert conv.depthwise_conv.norm_name == 'bn'
    assert conv.pointwise_conv.norm_name == 'bn'
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # add test for ['norm', 'conv', 'act']
    conv = DepthwiseSeparableConvModule(3, 8, 2, order=('norm', 'conv', 'act'))
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, with_spectral_norm=True)
    assert hasattr(conv.depthwise_conv.conv, 'weight_orig')
    assert hasattr(conv.pointwise_conv.conv, 'weight_orig')
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, padding_mode='reflect')
    assert isinstance(conv.depthwise_conv.padding_layer, nn.ReflectionPad2d)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # test dw_act_cfg
    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, dw_act_cfg=dict(type='LeakyReLU'))
    assert conv.depthwise_conv.activate.__class__.__name__ == 'LeakyReLU'
    assert conv.pointwise_conv.activate.__class__.__name__ == 'ReLU'
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # test pw_act_cfg
    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, pw_act_cfg=dict(type='LeakyReLU'))
    assert conv.depthwise_conv.activate.__class__.__name__ == 'ReLU'
    assert conv.pointwise_conv.activate.__class__.__name__ == 'LeakyReLU'
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # test act_cfg
    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, act_cfg=dict(type='LeakyReLU'))
    assert conv.depthwise_conv.activate.__class__.__name__ == 'LeakyReLU'
    assert conv.pointwise_conv.activate.__class__.__name__ == 'LeakyReLU'
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)
