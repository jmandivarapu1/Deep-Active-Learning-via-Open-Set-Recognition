import torch
import torch.nn as nn
# from torch.utils import load_state_dict_from_url
from torchvision import models
from collections import OrderedDict
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
    return model

def get_feat_size(block, spatial_size, ncolors=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.
    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    """

    x = torch.randn(2, ncolors, spatial_size, spatial_size)
    #print("block isss",block,ncolors, spatial_size, spatial_size)
    out = block(x)
    #print("block put",out)
    #print(out.size())
    num_feat = out.size(1)
    spatial_dim_x = out.size(2)
    spatial_dim_y = out.size(3)
    
    #sys.exit()
    return num_feat, spatial_dim_x, spatial_dim_y

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class Resenet(nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(Resenet, self).__init__()

        self.widen_factor = args.wrn_widen_factor
        self.depth = args.wrn_depth

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.dropout = args.dropout
        self.device = device

        self.nChannels = [16, 160, 320, 512]#[64,128,256,512]#[args.wrn_embedding_size, 16 * self.widen_factor, 32 * self.widen_factor,64 * self.widen_factor]

        #print("total nchannekls are",self.nChannels)

        assert ((self.depth - 4) % 6 == 0)
        self.num_block_layers = int((self.depth - 4) / 6)

        self.variational = False
        self.joint = False

        if args.train_var:
            self.variational = True
            self.num_samples = args.var_samples
            self.latent_dim = args.var_latent_dim

        if args.joint:
            self.joint = True
       
        
        #make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],batch_norm=True)
        #model=models.resnet18(pretrained=False)
        # del model.avgpool
        # del model.fc
        model=models.resnet18(pretrained=True)#=args.num_classes)
        self.encoder =  nn.Sequential(*list(model.children())[:-2])#nn.Sequential(*[model.features[i] for i in range(44)])#models.resnet18(pretrained=False,num_classes=512)#nn.Sequential(*[model.features[i] for i in range(44)])#resnet18(pretrained=False)#nn.Sequential(*[model for i in range(1)])

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,self.num_colors)
        if self.variational:
            self.latent_mu = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels,self.latent_dim, bias=False)
            self.latent_std = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,self.latent_dim, bias=False)
            self.latent_feat_out = self.latent_dim
        else:
            self.latent_feat_out = self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels
            self.latent_dim = self.latent_feat_out
            #print(self.latent_dim)

        if self.joint:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

            if self.variational:
                self.latent_decoder = nn.Linear(self.latent_feat_out, self.enc_spatial_dim_x * self.enc_spatial_dim_y *
                                                self.enc_channels, bias=False)

            self.decoder = nn.Sequential(OrderedDict([
                ('decoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[2],
                                                   WRNBasicBlock, dropout=self.dropout, batchnorm=self.batch_norm,
                                                   stride=1)),
                ('decoder_upsample1', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[1],
                                                   WRNBasicBlock, dropout=self.dropout, batchnorm=self.batch_norm,
                                                   stride=1)),
                ('decoder_upsample2', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[0],
                                                   WRNBasicBlock, dropout=self.dropout, batchnorm=self.batch_norm,
                                                   stride=1)),
                ('decoder_bn1', nn.BatchNorm2d(self.nChannels[0], eps=self.batch_norm)),
                ('decoder_act1', nn.ReLU(inplace=True)),
                ('decoder_conv1', nn.Conv2d(self.nChannels[0], self.num_colors, kernel_size=3, stride=1, padding=1,
                                            bias=False))
            ]))
        #      nn.Sequential(
        #     nn.Linear(128*512, 1024*4*4),                           # B, 1024*8*8
        #     View((-1, 1024, 4, 4)),                               # B, 1024,  8,  8
        #     nn.ConvTranspose2d(512, 512, 3, 1, 1, bias=False),   # B,  512, 16, 16
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(512, 256, 3, 1, 1, bias=False),    # B,  256, 32, 32
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 128, 3, 1, 1, bias=False),    # B,  128, 64, 64
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 3, 1),                       # B,   nc, 64, 64
        # )
            
        
        else:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

        # self._initialize_weights()

    def encode(self, x):
        
        x = self.encoder(x)
        z=x
        if self.variational:
            x = x.view(x.size(0), -1)
            z_mean = self.latent_mu(x)
            z_std = self.latent_std(x)
            return z_mean, z_std,z
        else:
            return x

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        if self.variational:
            z = self.latent_decoder(z)
            z = z.view(z.size(0), self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        x = self.decoder(z)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        if self.variational:
            z_mean, z_std,_ = self.encode(x)
            if self.joint:
                output_samples = torch.zeros(self.num_samples, x.size(0), self.num_colors, self.patch_size,
                                             self.patch_size).to(self.device)
                classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            else:
                output_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            for i in range(self.num_samples):
                z = self.reparameterize(z_mean, z_std)
                if self.joint:
                    output_samples[i] = self.decode(z)
                    classification_samples[i] = self.classifier(z)
                else:
                    output_samples[i] = self.classifier(z)
            if self.joint:
                return classification_samples, output_samples, z_mean, z_std
            else:
                return output_samples, z_mean, z_std
        else:
            x = self.encode(x)
            if self.joint:
                recon = self.decode(x)
                classification = self.classifier(x.view(x.size(0), -1))
                return classification, recon
            else:
                output = self.classifier(x.view(x.size(0), -1))
            return output

# print(resnet18(pretrained=False))