from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3, EdgeDetection, HED, CASENet, RCF
from .backbone import resnet
from .backbone import resnet_cbam
from .backbone import mobilenetv2

def _edge_rcf(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    backbone = resnet.__dict__[backbone_name](pretrained=pretrained_backbone)
    return_layers = {'layer4': 'layer4', 'layer3': 'layer3',
                     'layer2': 'layer2', 'layer1': 'layer1'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = RCF()
    model = EdgeDetection(backbone, classifier)

    return model

def _edge_case(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    backbone = resnet.__dict__[backbone_name](pretrained=pretrained_backbone)
    return_layers = {'layer4': 'layer4', 'layer3': 'layer3',
                     'layer2': 'layer2', 'layer1': 'layer1'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = CASENet()
    model = EdgeDetection(backbone, classifier)

    return model

def _edge_hed(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    backbone = resnet.__dict__[backbone_name](pretrained=pretrained_backbone)
    return_layers = {'layer4': 'layer4', 'layer3': 'layer3',
                     'layer2': 'layer2', 'layer1': 'layer1'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = HED()
    model = EdgeDetection(backbone, classifier)

    return model


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if backbone_name.endswith('cbam'):
        backbone = resnet_cbam.__dict__[backbone_name](
            replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'layer4', 'layer1': 'layer1'}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'layer4'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    # return_layers=return_layers
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)

    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(
        pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name == 'deeplabv3plus':
        return_layers = {'high_level_features': 'out',
                         'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if arch_type == 'rcf':
        model = _edge_rcf(arch_type, backbone, num_classes,
                        output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    elif arch_type == 'case':
        model = _edge_case(arch_type, backbone, num_classes,
                        output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    elif arch_type == 'hed':
        model = _edge_hed(arch_type, backbone, num_classes,
                          output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    elif backbone == 'mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes,
                                output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes,
                             output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    else:
        raise NotImplementedError

    return model
    

# Hed

def hed_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('hed', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

# CASE

def case_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('case', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

# RCF

def rcf_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('rcf', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

# Deeplab v3

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


# Deeplab v3+

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet50_cbam(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50_cbam', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101_cbam(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101_cbam', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
