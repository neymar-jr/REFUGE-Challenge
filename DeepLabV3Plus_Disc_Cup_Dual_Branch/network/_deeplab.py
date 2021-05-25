import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .attention import Attention_Assemble as A
from .utils import _SimpleSegmentationModel

__all__ = ["DeepLabV3", "EdgeDetection"]


class EdgeDetection(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, inner_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, padding=0),
        )
 
    def forward(self, x):
        return self.double_conv(x)

class DeepLabHeadV3Plus_RCF_Dual(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_RCF_Dual, self).__init__()

        self.netScoreOne = nn.Conv2d(256, 2, 1, stride=1, padding=0)
        self.netScoreTwo = nn.Conv2d(512, 2, 1, stride=1, padding=0)
        self.netScoreThr = nn.Conv2d(1024, 2, 1, stride=1, padding=0)
        self.netScoreFou = DoubleConv(2048, 64, 2)

        self.netCombine = nn.Sequential(
            nn.Conv2d(8, 2, 1)
        )

        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        # SENet
        # self.low_se = A.SELayer(in_dim=48, reduction=2, batch_first=True)
        # self.high_se = A.SELayer(in_dim=256, reduction=2, batch_first=True)

        # CBAM
        # self.low_ca = A.ChannelAttention(in_channels = 48, reduction = 4, batch_first = True)
        # self.low_pa = A.SpatialAttention(kernel_size = 3, batch_first = True)
        # self.high_ca = A.ChannelAttention(in_channels = 256, reduction = 4, batch_first = True)
        # self.high_pa = A.SpatialAttention(kernel_size = 3, batch_first = True)

        # ECANet
        # self.low_eca = A.ECA(channel=48)
        # self.high_eca = A.ECA(channel=256)

        # Non-local
        # self.low_nonlocal = A.NonLocalBlockND(in_channels=48)
        # self.high_nonlocal = A.NonLocalBlockND(in_channels=256)

        # GCNet
        # self.low_gc = A.GCNet_Atten(in_dim=48)
        # self.high_gc = A.GCNet_Atten(in_dim=256)

        # Triplet_Attention
        # self.low_tri = A.TripletAttention()
        # self.high_tri = A.TripletAttention()

        self._init_weight()

    def forward(self, feature):

        feature1 = feature['layer1']
        feature2 = feature['layer2']
        feature3 = feature['layer3']
        feature4 = feature['layer4']

        score1 = self.netScoreOne(feature1)
        score2 = self.netScoreTwo(feature2)
        score3 = self.netScoreThr(feature3)
        score4 = self.netScoreFou(feature4)
        
        sc_size = (feature1.shape[2]*4, feature1.shape[3]*4)

        score1 = F.interpolate(
            input=score1, size=sc_size, mode='bilinear', align_corners=False)
        score2 = F.interpolate(
            input=score2, size=sc_size, mode='bilinear', align_corners=False)
        score3 = F.interpolate(
            input=score3, size=sc_size, mode='bilinear', align_corners=False)
        score4 = F.interpolate(
            input=score4, size=sc_size, mode='bilinear', align_corners=False)

        out_dim = 8
        out_tensor = Variable(torch.FloatTensor(score1.size(
            0), out_dim, score1.size(2), score1.size(3))).cuda()

        class_num = 0
        for i in range(0, out_dim, 4):
            out_tensor[:, i, :, :] = score1[:, 0, :, :]
            out_tensor[:, i + 1, :, :] = score2[:, 0, :, :]
            out_tensor[:, i + 2, :, :] = score3[:, 0, :, :]
            out_tensor[:, i + 3, :, :] = score4[:, class_num, :, :]
            class_num += 1

        edge = self.netCombine(out_tensor)

        # low_level_feature: channel 256, down 4
        # output_feature：channel 2048, down 16
        low_level_feature = self.project(feature1)

        # low_level_feature = self.low_se(low_level_feature)
        # low_level_feature = self.low_ca(low_level_feature)
        # low_level_feature = self.low_pa(low_level_feature)
        # low_level_feature = self.low_eca(low_level_feature)
        # low_level_feature = self.low_nonlocal(low_level_feature)
        # low_level_feature = self.low_gc(low_level_feature)
        # low_level_feature = self.low_tri(low_level_feature)

        output_feature = self.aspp(feature4)

        # output_feature = self.high_se(output_feature)
        # output_feature = self.high_ca(output_feature)
        # output_feature = self.high_pa(output_feature)
        # output_feature = self.high_eca(output_feature)
        # output_feature = self.high_nonlocal(output_feature)
        # output_feature = self.high_gc(output_feature)
        # output_feature = self.high_tri(output_feature)

        output_feature = F.interpolate(
            input=output_feature, size=feature1.shape[2:], mode='bilinear', align_corners=False)

        # output_feature: channel 256, down 4
        seg = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        seg = F.interpolate(input=seg, size=sc_size, mode='bilinear', align_corners=False)
        return torch.cat([edge, seg], dim=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus_HED_Dual(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_HED_Dual, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        self.netScoreOne = nn.Conv2d(256, 2, 1, stride=1, padding=0)
        self.netScoreTwo = nn.Conv2d(512, 2, 1, stride=1, padding=0)
        self.netScoreThr = nn.Conv2d(1024, 2, 1, stride=1, padding=0)
        self.netScoreFou = nn.Conv2d(2048, 2, 1, stride=1, padding=0)

        self.netCombine = nn.Sequential(
            nn.Conv2d(8, 2, 1)
        )

        # SENet
        # self.low_se = A.SELayer(in_dim=48, reduction=2, batch_first=True)
        # self.high_se = A.SELayer(in_dim=256, reduction=2, batch_first=True)

        # CBAM
        # self.low_ca = A.ChannelAttention(in_channels = 48, reduction = 4, batch_first = True)
        # self.low_pa = A.SpatialAttention(kernel_size = 3, batch_first = True)
        # self.high_ca = A.ChannelAttention(in_channels = 256, reduction = 4, batch_first = True)
        # self.high_pa = A.SpatialAttention(kernel_size = 3, batch_first = True)

        # ECANet
        # self.low_eca = A.ECA(channel=48)
        # self.high_eca = A.ECA(channel=256)

        # Non-local
        # self.low_nonlocal = A.NonLocalBlockND(in_channels=48)
        # self.high_nonlocal = A.NonLocalBlockND(in_channels=256)

        # GCNet
        # self.low_gc = A.GCNet_Atten(in_dim=48)
        # self.high_gc = A.GCNet_Atten(in_dim=256)

        # Triplet_Attention
        # self.low_tri = A.TripletAttention()
        # self.high_tri = A.TripletAttention()

        self._init_weight()

    def forward(self, feature):

        feature1 = feature['layer1']
        feature2 = feature['layer2']
        feature3 = feature['layer3']
        feature4 = feature['layer4']

        sc_size = (feature1.shape[2]*4, feature1.shape[3]*4)

        # low_level_feature: channel 256, down 4
        # output_feature: channel 2048, down 16
        low_level_feature = self.project(feature1)


        # low_level_feature = self.low_se(low_level_feature)
        # low_level_feature = self.low_ca(low_level_feature)
        # low_level_feature = self.low_pa(low_level_feature)
        # low_level_feature = self.low_eca(low_level_feature)
        # low_level_feature = self.low_nonlocal(low_level_feature)
        # low_level_feature = self.low_gc(low_level_feature)
        # low_level_feature = self.low_tri(low_level_feature)

        output_feature = self.aspp(feature4)

        # output_feature = self.high_se(output_feature)
        # output_feature = self.high_ca(output_feature)
        # output_feature = self.high_pa(output_feature)
        # output_feature = self.high_eca(output_feature)
        # output_feature = self.high_nonlocal(output_feature)
        # output_feature = self.high_gc(output_feature)
        # output_feature = self.high_tri(output_feature)

        output_feature = F.interpolate(
            input=output_feature, size=feature1.shape[2:], mode='bilinear', align_corners=False)

        # output_feature: channel 256, down 4
        seg = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        seg = F.interpolate(input=seg, size=sc_size, mode='bilinear', align_corners=False)

        score1 = self.netScoreOne(feature1)
        score2 = self.netScoreTwo(feature2)
        score3 = self.netScoreThr(feature3)
        score4 = self.netScoreFou(feature4)

        score1 = F.interpolate(
            input=score1, size=sc_size, mode='bilinear', align_corners=False)
        score2 = F.interpolate(
            input=score2, size=sc_size, mode='bilinear', align_corners=False)
        score3 = F.interpolate(
            input=score3, size=sc_size, mode='bilinear', align_corners=False)
        score4 = F.interpolate(
            input=score4, size=sc_size, mode='bilinear', align_corners=False)

        edge = self.netCombine(torch.cat([score1, score2, score3, score4], dim=1))

        return torch.cat([edge, seg], dim=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus_CASE_Dual(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_CASE_Dual, self).__init__()

        self.netScoreOne = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.netScoreTwo = nn.Conv2d(512, 1, 1, stride=1, padding=0)
        self.netScoreThr = nn.Conv2d(1024, 1, 1, stride=1, padding=0)
        self.netScoreFou = nn.Conv2d(2048, 2, 1, stride=1, padding=0)

        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.netCombine = nn.Sequential(
            nn.Conv2d(8, 2, 1)
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        # SENet
        # self.low_se = A.SELayer(in_dim=48, reduction=2, batch_first=True)
        # self.high_se = A.SELayer(in_dim=256, reduction=2, batch_first=True)

        # CBAM
        # self.low_ca = A.ChannelAttention(in_channels = 48, reduction = 4, batch_first = True)
        # self.low_pa = A.SpatialAttention(kernel_size = 3, batch_first = True)
        # self.high_ca = A.ChannelAttention(in_channels = 256, reduction = 4, batch_first = True)
        # self.high_pa = A.SpatialAttention(kernel_size = 3, batch_first = True)

        # ECANet
        # self.low_eca = A.ECA(channel=48)
        # self.high_eca = A.ECA(channel=256)

        # Non-local
        # self.low_nonlocal = A.NonLocalBlockND(in_channels=48)
        # self.high_nonlocal = A.NonLocalBlockND(in_channels=256)

        # GCNet
        # self.low_gc = A.GCNet_Atten(in_dim=48)
        # self.high_gc = A.GCNet_Atten(in_dim=256)

        # Triplet_Attention
        # self.low_tri = A.TripletAttention()
        # self.high_tri = A.TripletAttention()

        self._init_weight()

    def forward(self, feature):

        feature1 = feature['layer1']
        feature2 = feature['layer2']
        feature3 = feature['layer3']
        feature4 = feature['layer4']

        score1 = self.netScoreOne(feature1)
        score2 = self.netScoreTwo(feature2)
        score3 = self.netScoreThr(feature3)
        score4 = self.netScoreFou(feature4)
        
        sc_size = (feature1.shape[2]*4, feature1.shape[3]*4)

        score1 = F.interpolate(
            input=score1, size=sc_size, mode='bilinear', align_corners=False)
        score2 = F.interpolate(
            input=score2, size=sc_size, mode='bilinear', align_corners=False)
        score3 = F.interpolate(
            input=score3, size=sc_size, mode='bilinear', align_corners=False)
        score4 = F.interpolate(
            input=score4, size=sc_size, mode='bilinear', align_corners=False)

        out_dim = 8
        out_tensor = Variable(torch.FloatTensor(score1.size(
            0), out_dim, score1.size(2), score1.size(3))).cuda()

        class_num = 0
        for i in range(0, out_dim, 4):
            out_tensor[:, i, :, :] = score1[:, 0, :, :]
            out_tensor[:, i + 1, :, :] = score2[:, 0, :, :]
            out_tensor[:, i + 2, :, :] = score3[:, 0, :, :]
            out_tensor[:, i + 3, :, :] = score4[:, class_num, :, :]
            class_num += 1

        edge = self.netCombine(out_tensor)

        # low_level_feature: channel 256, down 4
        # output_feature：channel 2048, down 16
        low_level_feature = self.project(feature1)

        # low_level_feature = self.low_se(low_level_feature)
        # low_level_feature = self.low_ca(low_level_feature)
        # low_level_feature = self.low_pa(low_level_feature)
        # low_level_feature = self.low_eca(low_level_feature)
        # low_level_feature = self.low_nonlocal(low_level_feature)
        # low_level_feature = self.low_gc(low_level_feature)
        # low_level_feature = self.low_tri(low_level_feature)

        output_feature = self.aspp(feature4)

        # output_feature = self.high_se(output_feature)
        # output_feature = self.high_ca(output_feature)
        # output_feature = self.high_pa(output_feature)
        # output_feature = self.high_eca(output_feature)
        # output_feature = self.high_nonlocal(output_feature)
        # output_feature = self.high_gc(output_feature)
        # output_feature = self.high_tri(output_feature)

        output_feature = F.interpolate(
            input=output_feature, size=feature1.shape[2:], mode='bilinear', align_corners=False)

        # output_feature: channel 256, down 4
        seg = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        seg = F.interpolate(input=seg, size=sc_size, mode='bilinear', align_corners=False)
        return torch.cat([edge, seg], dim=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

        # SENet
        # self.low_se = A.SELayer(in_dim=48, reduction=2, batch_first=True)
        # self.high_se = A.SELayer(in_dim=256, reduction=2, batch_first=True)

        # CBAM
        # self.low_ca = A.ChannelAttention(in_channels = 48, reduction = 4, batch_first = True)
        # self.low_pa = A.SpatialAttention(kernel_size = 3, batch_first = True)
        # self.high_ca = A.ChannelAttention(in_channels = 256, reduction = 4, batch_first = True)
        # self.high_pa = A.SpatialAttention(kernel_size = 3, batch_first = True)

        # ECANet
        # self.low_eca = A.ECA(channel=48)
        # self.high_eca = A.ECA(channel=256)

        # Non-local
        # self.low_nonlocal = A.NonLocalBlockND(in_channels=48)
        # self.high_nonlocal = A.NonLocalBlockND(in_channels=256)

        # GCNet
        # self.low_gc = A.GCNet_Atten(in_dim=48)
        # self.high_gc = A.GCNet_Atten(in_dim=256)

        # Triplet_Attention
        # self.low_tri = A.TripletAttention()
        # self.high_tri = A.TripletAttention()

    def forward(self, feature):
        # low_level_feature from stage1: channel 256, down 4
        # output_feature from stage4：channel 2048, down 16
        feature1 = feature['layer1']
        feature4 = feature['layer4']

        low_level_feature = self.project(feature1)
        # low_level_feature: channel 48, down 4

        # low_level_feature = self.low_se(low_level_feature)
        # low_level_feature = self.low_ca(low_level_feature)
        # low_level_feature = self.low_pa(low_level_feature)
        # low_level_feature = self.low_eca(low_level_feature)
        # low_level_feature = self.low_nonlocal(low_level_feature)
        # low_level_feature = self.low_gc(low_level_feature)

        output_feature = self.aspp(feature4)
        # ouput_feature: channel 256, down 16

        # output_feature = self.high_se(output_feature)
        # output_feature = self.high_ca(output_feature)
        # output_feature = self.high_pa(output_feature)
        # output_feature = self.high_eca(output_feature)
        # output_feature = self.high_nonlocal(output_feature)
        # output_feature = self.high_gc(output_feature)

        output_feature = F.interpolate(
            output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        # output_feature: channel 256, down 4

        sc_size = (feature1.shape[2]*4, feature1.shape[3]*4)
        seg = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        seg = F.interpolate(input=seg, size=sc_size, mode='bilinear', align_corners=False)

        return seg

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['layer4'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


if __name__ == '__main__':
    low_feature = torch.randn(size=(8, 48, 128, 128))
    high_feature = torch.randn(size=(8, 256, 32, 32))

    # high_feature = high_ca(high_feature)
    # high_feature = high_pa(high_feature)

    print(low_feature.size())
    print(high_feature.size())