import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F

import numpy as np

from .multitask_model import MultiTaskModel

from typing import Iterator
import itertools

class FPN(nn.Module, MultiTaskModel):
    def __init__(self, num_classes, backbone='resnet34', use_depth=False, use_custom_fpn=False, add_output_exp=False, init_decoder_gain=0.1, fpn_args={}, pretrained=True, in_channels=None, use_only_depth=True):
        super().__init__()

        self.use_custom_fpn = use_custom_fpn
        print('Creating FPN segmentation with {} classes'.format(num_classes))
        if 'encoder_depth' not in fpn_args:
            fpn_args['encoder_depth'] = 4
        if 'upsampling' not in fpn_args:
            fpn_args['upsampling'] = 2

        self.use_only_depth = use_only_depth

        if use_depth:
            if not use_only_depth:
                self.depth_mean = fpn_args['depth_mean']
                self.depth_std = fpn_args['depth_std']
        fpn_args.pop('depth_mean')
        fpn_args.pop('depth_std')
            # else:

        

        self.use_depth = use_depth
        self.in_channels = in_channels if in_channels is not None else (4 if use_depth else 3)

        if self.use_custom_fpn:
            self.model = Custom_FPN(backbone, classes=sum(num_classes), activation=None, in_channels=self.in_channels, decoder_merge_policy='add', encoder_weights='imagenet' if pretrained else None, **fpn_args)
        else:
            self.model = smp.FPN(backbone, classes=sum(num_classes), activation=None,
                                 decoder_merge_policy='add', encoder_weights='imagenet' if pretrained else None, **fpn_args)

        self.preprocess_params = smp.encoders.get_preprocessing_params(backbone, pretrained="imagenet")
        if use_depth:
            if not use_only_depth:
                if type(self.depth_mean)==int or type(self.depth_mean)==float: # if one number, add to mean and std vectors
                    self.preprocess_params['mean'].append(self.depth_mean)
                    self.preprocess_params['std'].append(self.depth_std)
                else: # if list, extend the vectors
                    self.preprocess_params['mean'].extend(self.depth_mean)
                    self.preprocess_params['std'].extend(self.depth_std)
            else:
                # self.preprocess_params['mean'] = [0,0,0]
                # self.preprocess_params['std'] = [1,1,1]
                self.preprocess_params['mean'] = [0]
                self.preprocess_params['std'] = [1]

        # print("use_depth", use_depth)
        # print("use_only_depth", use_only_depth)
        # print(self.preprocess_params['mean'])

        self.add_output_exp = add_output_exp
        self.init_decoder_gain = init_decoder_gain

        if self.add_output_exp:
            output_exp_scale_var = torch.ones(size=(sum(num_classes),), dtype=torch.float32, requires_grad=True)
            self.output_exp_scale = torch.nn.Parameter(output_exp_scale_var)
            self.register_parameter('output_exp_scale',self.output_exp_scale)

    def init_output(self, num_vector_fields=1):
        with torch.no_grad():
            decoder = self.model.decoder

            decoder_convs = []

            if self.use_custom_fpn:
                for head in self.model.segmentation_head_list if len(self.model.segmentation_head_list) > 0 else [self.model.segmentation_head]:
                    output_conv = head[0]
                    decoder_convs += [output_conv.block[0], head[2]]

                decoder_convs += [decoder.p2.skip_conv, decoder.p3.skip_conv, decoder.p4.skip_conv, decoder.p5]
            else:
                output_conv = self.model.segmentation_head[0]
                decoder_convs += [output_conv]
                decoder_convs += [decoder.p2.skip_conv, decoder.p3.skip_conv, decoder.p4.skip_conv, decoder.p5]

            for seg_block in decoder.seg_blocks:
                for conv_block in seg_block.block:
                    decoder_convs.append(conv_block.block[0])

            for c in decoder_convs:
                if type(c) == torch.nn.modules.conv.Conv2d:
                    print('initialize decoder layer with size: ', c.weight.size())
                    torch.nn.init.xavier_normal_(c.weight,gain=self.init_decoder_gain)
                    if c.bias is not None:
                        torch.nn.init.zeros_(c.bias)


    def forward(self, input, only_encode=False):
        input = preprocess_input(input,**self.preprocess_params)
        output = self.model.forward(input)
        if self.add_output_exp:
            output = torch.stack((output[:,0],output[:,1],
                                  torch.exp(torch.exp(self.output_exp_scale[2])*output[:,2])-1,
                                  output[:,3]),dim=1)

        return output

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters shared by all tasks.
        Returns
        -------
        """
        if isinstance(self.model, MultiTaskModel):
            return self.model.shared_parameters()
        else:
            # encode and decode parameters are shared
            encoder_decoder_params = [self.encoder.parameters(), self.decoder.parameters()]

            # only the last convolution/linear layer in segmentation head is task specific
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = _find_last_conv_index(self.segmentation_head)

            return itertools.chain(*[h.parameters() for h in self.segmentation_head[last_conv_layer_idx:]],
                                   *encoder_decoder_params)

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters specific to each task.
        Returns
        -------
        """
        if isinstance(self.model, MultiTaskModel):
            return self.model.task_specific_parameters()
        else:
            # only the last convolution/linear layer in segmentation head is task specific
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = _find_last_conv_index(self.model.segmentation_head)

            return itertools.chain(*[h.parameters() for h in self.segmentation_head[last_conv_layer_idx:]])

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        if isinstance(self.model, MultiTaskModel):
            return self.model.last_shared_parameters()
        else:
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = self._find_last_conv_index(self.segmentation_head)

            # last shared parameters ar all in the segmentation head before last convolution layer
            return itertools.chain(*[h.parameters() for h in self.segmentation_head[:last_conv_layer_idx]])


def _find_last_conv_index(sequence):
    # find last convolution layer in segmentation head
    return max([i for i, layer in enumerate(sequence)
                               if type(layer) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d,
                                                  nn.ConvTranspose1d, nn.Conv3d, nn.ConvTranspose3d]])

from typing import Optional, Union
try:
    # older version <=v0.2.1
    from segmentation_models_pytorch.fpn.decoder import FPNBlock, MergeBlock, Conv3x3GNReLU, FPNDecoder
except:
    # newer version >=v0.3.0
    from segmentation_models_pytorch.decoders.fpn.decoder import FPNBlock, MergeBlock, Conv3x3GNReLU, FPNDecoder
from segmentation_models_pytorch.base import SegmentationModel, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.base.initialization import initialize_head

class Custom_FPN(SegmentationModel,MultiTaskModel):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
        decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
        decoder_merge_policy: determines how to merge outputs inside FPN.
            One of [``add``, ``cat``]
        decoder_dropout: spatial dropout rate in range (0, 1).
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation (str, callable): activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax2d``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_segmentation_head_channels: int = 64,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 2,
        aux_params: Optional[dict] = None,
        checkpoint_encoder_features: bool = False,
        classes_grouping: Optional[list] = None,

    ):
        super().__init__()

        print("in_channels FPN", in_channels)

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head_list = nn.ModuleList()
        self.classes_grouping = classes_grouping
        if not self.classes_grouping:
            self.segmentation_head = SegmentationHead(
                in_channels=self.decoder.out_channels,
                segmentation_channels=decoder_segmentation_head_channels,
                out_channels=classes,
                activation=activation,
                kernel_size=3,
                upsampling=upsampling,
            )
        else:
            # dummy head for compatibility with SegmentationModel
            self.segmentation_head = nn.Sequential(nn.Identity())

            for group_members in self.classes_grouping:
                num_group_members = len(group_members) if type(group_members) in [list,tuple] else group_members
                self.segmentation_head_list.append(SegmentationHead(
                    in_channels=self.decoder.out_channels,
                    segmentation_channels=decoder_segmentation_head_channels,
                    out_channels=num_group_members,
                    activation=activation,
                    kernel_size=3,
                    upsampling=upsampling,
                ))

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.checkpoint_encoder_features = checkpoint_encoder_features
        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        super().initialize()

        for seg_head in self.segmentation_head_list:
            initialize_head(seg_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)

        if self.checkpoint_encoder_features and any([f.requires_grad for f in features]):
            decoder_output = torch.utils.checkpoint.checkpoint(self.decoder, *features)
        else:
            decoder_output = self.decoder(*features)

        if self.classes_grouping is not None:
            assert len(self.segmentation_head_list) == len(self.classes_grouping)
            # concatenate all segmentation heads and permute them back to original order based on self.classes_grouping
            masks = torch.cat([head(decoder_output) for head in self.segmentation_head_list], dim=1)
            new_idx = list(np.concatenate(self.classes_grouping))
            old_idx = [new_idx.index(i) for i in range(len(new_idx))]
            masks = masks[:,old_idx]
        else:
            masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters shared by all tasks.
        Returns
        -------
        """
        # encode and decode parameters are shared
        encoder_decoder_params = [self.encoder.parameters(), self.decoder.parameters()]

        if self.classes_grouping is None:
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = _find_last_conv_index(self.segmentation_head)

            # add all parameters except for the last convolution layer to shared parameters
            encoder_decoder_params.append(self.segmentation_head[:last_conv_layer_idx].parameters())

        return itertools.chain(*encoder_decoder_params)

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters specific to each task.
        Returns
        -------
        """
        if self.classes_grouping is None:
            # only the last convolution/linear layer in segmentation head is task specific
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = _find_last_conv_index(self.segmentation_head)

            return itertools.chain(*[h.parameters() for h in self.segmentation_head[last_conv_layer_idx:]])
        else:
            # each segmentation head is task specific
            return itertools.chain(*[h.parameters() for h in self.segmentation_head_list])



    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        if self.classes_grouping is None:
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = _find_last_conv_index(self.segmentation_head)

            # last shared parameters ar all in the segmentation head before last convolution layer
            return itertools.chain(*[h.parameters() for h in self.segmentation_head[:last_conv_layer_idx]])
        else:
            # all decoder layers are last shared parameters
            return self.decoder.parameters()
#
#
# class FPNDecoder(nn.Module):
#     def __init__(
#             self,
#             encoder_channels,
#             encoder_depth=5,
#             pyramid_channels=256,
#             segmentation_channels=128,
#             dropout=0.2,
#             merge_policy="add",
#     ):
#         super().__init__()
#
#         self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
#         if encoder_depth < 3:
#             raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))
#
#         encoder_channels = encoder_channels[::-1]
#         encoder_channels = encoder_channels[:encoder_depth + 1]
#
#         self.conv1x1_c5 = nn.Conv2d(encoder_channels[0], pyramid_channels,kernel_size=1) if pyramid_channels != encoder_channels[0] else nn.Identity()
#
#         self.p5 = FPNBlock(pyramid_channels, encoder_channels[1])
#         self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
#         self.p4 = FPNBlock(pyramid_channels, encoder_channels[2])
#         self.p3 = FPNBlock(pyramid_channels, encoder_channels[3])
#         self.p2 = FPNBlock(pyramid_channels, encoder_channels[4])
#
#         self.seg_blocks = nn.ModuleList([
#             SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
#             for n_upsamples in [3, 2, 1, 0]
#         ])
#
#         self.merge = MergeBlock(merge_policy)
#         self.dropout = nn.Dropout2d(p=dropout, inplace=True)
#
#     def forward(self, *features):
#         c1, c2, c3, c4, c5 = features[-5:]
#
#         # convert number of output features to desired value
#         c5 = self.conv1x1_c5(c5)
#
#         # combine features in pyramid
#         p5 = self.p5(c5, c4)
#         p4 = self.p4(p5, c3)
#         p3 = self.p3(p4, c2)
#         p2 = self.p2(p3, c1)
#
#         feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
#         x = self.merge(feature_pyramid)
#         x = self.dropout(x)
#
#         return x
#
# class SegmentationBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, n_upsamples=0):
#         super().__init__()
#
#         blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=False),
#                   Conv3x3GNReLU(out_channels, out_channels, upsample=False)]
#
#         self.n_upsamples = n_upsamples
#         #if n_upsamples > 1:
#         #    for _ in range(1, n_upsamples):
#         #        blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
#
#         self.block = nn.Sequential(*blocks)
#
#     def forward(self, x):
#         x = self.block(x)
#         return F.interpolate(x, scale_factor=2**self.n_upsamples, mode="nearest") if self.n_upsamples > 0 else x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels,  segmentation_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d_1 = Conv3x3GNReLU(in_channels, segmentation_channels, upsample=False)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d_2 = nn.Conv2d(segmentation_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = Activation(activation)

        super().__init__(conv2d_1, upsampling, conv2d_2, activation)


def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = torch.tensor(mean)
        mean = mean.reshape([1, -1, 1, 1])
        x = x - mean.to(x.device)

    if std is not None:
        std = torch.tensor(std)
        std = std.reshape([1, -1, 1, 1])
        x = x / std.to(x.device)

    return x