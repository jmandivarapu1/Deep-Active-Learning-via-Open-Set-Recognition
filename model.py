import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from torchvision import models
import arguments_yaml
import yaml
from easydict import EasyDict
class MetaModel(nn.Module):
    def __init__(self):
        super(MetaModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*2*2)),                                 # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*4*4),                           # B, 1024*8*8
            View((-1, 1024, 4, 4)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)



###############################################################
#              CIFAR---CAE -CONTACITED AUTO ENCODER
###############################################################
class Discriminator1(nn.Module):
    def __init__(self, z_dim=10):
        super(Discriminator1, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512,bias = False)
        self.fc2 = nn.Linear(512,512,bias = False)
        self.fc3 = nn.Linear(512,1,bias = False)
        self.activation = nn.Tanh()
        self.sigmoid

    def encoder(self, x):
        h1 = self.sigmoid(self.fc1(x.view(-1, 61326)))
        return self.sigmoid(self.fc2(h1))

    def decoder(self, z):
        h2 = self.sigmoid(self.fc3(z))
        return self.sigmoid(self.fc4(h2))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

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
    num_feat = out.size(1)
    spatial_dim_x = out.size(2)
    spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y


class SingleConvLayer(nn.Module):
    """
    Convenience function defining a single block consisting of a convolution or transposed convolution followed by
    batch normalization and a rectified linear unit activation function.
    """
    def __init__(self, l, fan_in, fan_out, kernel_size=3, padding=1, stride=1, batch_norm=1e-5, dropout=0.0,
                 is_transposed=False):
        super(SingleConvLayer, self).__init__()

        if is_transposed:
            self.layer = nn.Sequential(OrderedDict([
                ('transposed_conv' + str(l), nn.ConvTranspose2d(fan_in, fan_out, kernel_size=kernel_size,
                                                                padding=padding, stride=stride, bias=False))
            ]))
        else:
            self.layer = nn.Sequential(OrderedDict([
                ('conv' + str(l), nn.Conv2d(fan_in, fan_out, kernel_size=kernel_size, padding=padding, stride=stride,
                                            bias=False))
            ]))

        if batch_norm > 0.0:
            self.layer.add_module('bn' + str(l), nn.BatchNorm2d(fan_out, eps=batch_norm))

        self.layer.add_module('act' + str(l), nn.ReLU())

        if not dropout == 0.0:
            self.layer.add_module('dropout', nn.Dropout2d(p=dropout, inplace=False))

    def forward(self, x):
        x = self.layer(x)
        return x


class SingleLinearLayer(nn.Module):
    """
    Convenience function defining a single block consisting of a fully connected (linear) layer followed by
    batch normalization and a rectified linear unit activation function.
    """
    def __init__(self, l, fan_in, fan_out, batch_norm=1e-5, dropout=0.0):
        super(SingleLinearLayer, self).__init__()

        self.fclayer = nn.Sequential(OrderedDict([
            ('fc' + str(l), nn.Linear(fan_in, fan_out, bias=False)),
        ]))

        if batch_norm > 0.0:
            self.fclayer.add_module('bn' + str(l), nn.BatchNorm1d(fan_out, eps=batch_norm))

        self.fclayer.add_module('act' + str(l), nn.ReLU())

        if not dropout == 0.0:
            self.fclayer.add_module('dropout', nn.Dropout2d(p=dropout, inplace=False))

    def forward(self, x):
        x = self.fclayer(x)
        return x


class MLP(nn.Module):
    """
    MLP design with two hidden layers and 400 hidden units each in the encoder according to
    ﻿Measuring Catastrophic Forgetting in Neural Networks: https://arxiv.org/abs/1708.02072
    Extended to the variational setting and our unified model.
    """

    def __init__(self, device, num_classes, num_colors, args):
        super(MLP, self).__init__()

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.device = device

        self.variational = False
        self.joint = False

        if args.train_var:
            self.variational = True
            self.num_samples = args.var_samples
            self.latent_dim = args.var_latent_dim
        else:
            self.latent_dim = 400

        if args.joint:
            self.joint = True

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_layer1', SingleLinearLayer(1, self.num_colors * (self.patch_size ** 2), 400,
                                                 batch_norm=self.batch_norm, dropout=self.dropout)),
            ('encoder_layer2', SingleLinearLayer(2, 400, 400, batch_norm=self.batch_norm, dropout=self.dropout))
        ]))

        if self.variational:
            self.latent_mu = nn.Linear(400, self.latent_dim, bias=False)
            self.latent_std = nn.Linear(400, self.latent_dim, bias=False)

        if self.joint:
            self.classifier = nn.Sequential(nn.Linear(self.latent_dim, num_classes, bias=False))

            if self.variational:
                self.latent_decoder = SingleLinearLayer(0, self.latent_dim, 400, batch_norm=self.batch_norm)

            self.decoder = nn.Sequential(OrderedDict([
                ('decoder_layer1', SingleLinearLayer(1, 400, 400, batch_norm=self.batch_norm, dropout=self.dropout)),
                ('decoder_layer2', nn.Linear(400, self.num_colors * (self.patch_size ** 2), bias=False))
            ]))
        else:
            self.classifier = nn.Sequential(nn.Linear(self.latent_dim, num_classes, bias=False))

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        if self.variational:
            z_mean = self.latent_mu(x)
            z_std = self.latent_std(x)
            return z_mean, z_std
        else:
            return x

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        if self.variational:
            z = self.latent_decoder(z)
        x = self.decoder(z)
        x = x.view(-1, self.num_colors, self.patch_size, self.patch_size)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        x = x.view(-1, self.num_colors, self.patch_size, self.patch_size)
        return x

    def forward(self, x):
        if self.variational:
            z_mean, z_std = self.encode(x)
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
                classification = self.classifier(x)
                return classification, recon
            else:
                output = self.classifier(x)
            return output


class DCNN(nn.Module):
    """
    CNN architecture inspired by WAE-DCGAN from https://arxiv.org/pdf/1511.06434.pdf but without the GAN component.
    Extended to the variational setting.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(DCNN, self).__init__()

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.dropout = args.dropout
        self.device = device

        # for 28x28 images, e.g. MNIST. We set the innermost convolution's kernel from 4 to 3 and adjust the
        # paddings in the decoder to upsample correspondingly. This way the incoming spatial dimensionality
        # to the latent space stays the same as with 32x32 resolution
        self.inner_kernel_size = 4
        self.inner_padding = 0
        self.outer_padding = 1
        if args.patch_size < 32:
            self.inner_kernel_size = 3
            self.inner_padding = 1
            self.outer_padding = 0

        self.variational = False
        self.joint = False

        if args.train_var:
            self.variational = True
            self.num_samples = args.var_samples
            self.latent_dim = args.var_latent_dim
        else:
            self.latent_dim = 1024

        if args.joint:
            self.joint = True

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_layer1', SingleConvLayer(1, self.num_colors, 128, kernel_size=4, stride=2, padding=1,
                                               batch_norm=self.batch_norm, dropout=self.dropout)),
            ('encoder_layer2', SingleConvLayer(2, 128, 256, kernel_size=4, stride=2, padding=1,
                                               batch_norm=self.batch_norm, dropout=self.dropout)),
            ('encoder_layer3', SingleConvLayer(3, 256, 512, kernel_size=4, stride=2, padding=1,
                                               batch_norm=self.batch_norm, dropout=self.dropout)),
            ('encoder_layer4', SingleConvLayer(4, 512, 1024, kernel_size=self.inner_kernel_size, stride=2, padding=0,
                                               batch_norm=self.batch_norm, dropout=self.dropout))
        ]))

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,
                                                                                          self.num_colors)
        if self.variational:
            self.latent_mu = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,
                                       self.latent_dim, bias=False)
            self.latent_std = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,
                                        self.latent_dim, bias=False)
            self.latent_feat_out = self.latent_dim
        else:
            self.latent_feat_out = self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels

        if self.joint:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

            if self.variational:
                self.latent_decoder = SingleLinearLayer(0, self.latent_feat_out, self.enc_spatial_dim_x *
                                                        self.enc_spatial_dim_y * self.enc_channels,
                                                        batch_norm=self.batch_norm)

            self.decoder =nn.Sequential(OrderedDict([
                ('decoder_layer1', SingleConvLayer(1, 1024, 512, kernel_size=4, stride=2, padding=self.inner_padding,
                                                   batch_norm=self.batch_norm, is_transposed=True,
                                                   dropout=self.dropout)),
                ('decoder_layer2', SingleConvLayer(2, 512, 256, kernel_size=4, stride=2, padding=self.outer_padding,
                                                   batch_norm=self.batch_norm, is_transposed=True,
                                                   dropout=self.dropout)),
                ('decoder_layer3', SingleConvLayer(3, 256, 128, kernel_size=4, stride=2, padding=self.outer_padding,
                                                   batch_norm=self.batch_norm, is_transposed=True,
                                                   dropout=self.dropout)),
                ('decoder_layer4', nn.ConvTranspose2d(128, self.num_colors, kernel_size=4, stride=2,
                                                      padding=1, bias=False))
            ]))
        else:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        if self.variational:
            z_mean = self.latent_mu(x)
            z_std = self.latent_std(x)
            return z_mean, z_std
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

    def forward(self, x):
        if self.variational:
            z_mean, z_std = self.encode(x)
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


class WRNBasicBlock(nn.Module):
    """
    Convolutional block consisting of multiple 3x3 convolutions with short-cuts,
    ReLU activation functions and batch normalization.
    """
    def __init__(self, in_planes, out_planes, stride, batchnorm=1e-5, dropout=0.0, is_transposed=False):
        super(WRNBasicBlock, self).__init__()

        self.p_drop = dropout

        if not self.p_drop == 0.0:
            self.dropout = nn.Dropout2d(p=dropout, inplace=False)

        # TODO: hard-coded kernel size, padding/out-padding may work only for width X height: 8 X 8, 16 x 16 etc.
        if is_transposed:
            self.layer1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                             output_padding=int(stride > 1), bias=False)
        else:
            self.layer1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.useShortcut = ((in_planes == out_planes) and (stride == 1))
        if not self.useShortcut:
            if is_transposed:
                self.shortcut = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                                                   output_padding=int(1 and stride == 2), bias=False)
            else:
                self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        if not self.useShortcut:
            if self.p_drop == 0.0:
                x = self.relu1(self.bn1(x))
            else:
                x = self.dropout(self.relu1(self.bn1(x)))
        else:
            if self.p_drop == 0.0:
                out = self.relu1(self.bn1(x))
            else:
                out = self.dropout(self.relu1(self.bn1(x)))

        if self.p_drop == 0.0:
            out = self.relu2(self.bn2(self.layer1(out if self.useShortcut else x)))
        else:
            out = self.relu2(self.bn2(self.dropout(self.layer1(out if self.useShortcut else x))))

        if not self.p_drop == 0.0:
            out = self.dropout(out)

        out = self.conv2(out)

        return torch.add(x if self.useShortcut else self.shortcut(x), out)


class WRNNetworkBlock(nn.Module):
    """
    Convolutional or transposed convolutional block
    """
    def __init__(self, nb_layers, in_planes, out_planes, block_type, batchnorm=1e-5, stride=1, dropout=0.0,
                 is_transposed=False):
        super(WRNNetworkBlock, self).__init__()

        if is_transposed:
            self.block = nn.Sequential(OrderedDict([
                ('deconv_block' + str(layer + 1), block_type(layer == 0 and in_planes or out_planes, out_planes,
                                                             layer == 0 and stride or 1, dropout, batchnorm=batchnorm,
                                                             is_transposed=(layer == 0), dropout=dropout))
                for layer in range(nb_layers)
            ]))
        else:
            self.block = nn.Sequential(OrderedDict([
                ('conv_block' + str(layer + 1), block_type((layer == 0 and in_planes) or out_planes, out_planes,
                                                           (layer == 0 and stride) or 1,
                                                           dropout=dropout, batchnorm=batchnorm))
                for layer in range(nb_layers)
            ]))

    def forward(self, x):
        x = self.block(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_vgg():
    task_model=models.vgg16(pretrained=True)
    del task_model.classifier
    del task_model.avgpool

    return task_model

class WRN(nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(WRN, self).__init__()

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
        
        self.encoder = make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],batch_norm=True)
        # nn.Sequential(OrderedDict([
        #     ('encoder_conv1', nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, dropout=self.dropout)),
        #     ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, stride=2,
        #                                        dropout=self.dropout)),
        #     ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, stride=2,
        #                                        dropout=self.dropout)),
        #     ('encoder_bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.batch_norm)),
        #     ('encoder_act1', nn.ReLU(inplace=True))
        # ]))

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

            self.decoder =      nn.Sequential(OrderedDict([
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
        if self.variational:
            x = x.view(x.size(0), -1)
            z_mean = self.latent_mu(x)
            z_std = self.latent_std(x)
            return z_mean, z_std
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
            z_mean, z_std = self.encode(x)
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


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class WRN_actual(nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(WRN_actual, self).__init__()

        self.widen_factor = args.wrn_widen_factor
        self.depth = args.wrn_depth

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.dropout = args.dropout
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.nChannels = [args.wrn_embedding_size, 16 * self.widen_factor, 32 * self.widen_factor,512]
                          #64 * self.widen_factor]

        #print("total nchannekls are",self.nChannels,self)
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
        #print("total nchannekls are",self.nChannels,self.num_block_layers, self.variational,self.num_samples,self.latent_dim,self.batch_norm )
        #make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],batch_norm=True)
        model=models.vgg16_bn(num_classes=args.num_classes)
        self.encoder = nn.Sequential(*[model.features[i] for i in range(44)])#make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],batch_norm=True)
        #nn.Sequential(*[model.features[i] for i in range(31)]) #
        
        # nn.Sequential(OrderedDict([
        #     ('encoder_conv1', nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, dropout=self.dropout)),
        #     ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, stride=2,
        #                                        dropout=self.dropout)),
        #     ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, stride=2,
        #                                        dropout=self.dropout)),
        #     ('encoder_bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.batch_norm)),
        #     ('encoder_act1', nn.ReLU(inplace=True))
        # ]))

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,
                                                                                          self.num_colors)
        if self.variational:
            self.latent_mu = nn.Linear(512,self.latent_dim, bias=False)#self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels32768
            self.latent_std = nn.Linear(512,self.latent_dim, bias=False)#self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,32768
            self.latent_feat_out = self.latent_dim
        else:
            self.latent_feat_out = self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels
            self.latent_dim = self.latent_feat_out
            #print(self.latent_dim)

        if self.joint:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

            if self.variational:
                self.latent_decoder = nn.Linear(self.latent_feat_out,32768)
                #self.enc_spatial_dim_x * self.enc_spatial_dim_y *self.enc_channels, bias=False)

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
        else:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

    def encode(self, x):
        x = self.encoder(x)
        classifier_z = self.avgpool(x)
        #x = self.avgpool(x)
        #print("shape of encoder is",x.shape)
        if self.variational:
            x = x.view(x.size(0), -1)
            z_mean = self.latent_mu(x)
            z_std = self.latent_std(x)
            return z_mean, z_std,classifier_z
        else:
            return x

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        if self.variational:
            z = self.latent_decoder(z)
            z = z.view(z.size(0),512,8,8 )#self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        #print("z.size is",z.shape)
        x = self.decoder(z)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        if self.variational:
            z_mean, z_std,classifier_z = self.encode(x)
            if self.joint:
                output_samples = torch.zeros(self.num_samples, x.size(0), self.num_colors, self.patch_size,
                                             self.patch_size).to(self.device)
                classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            else:
                output_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            for i in range(self.num_samples):
                z = self.reparameterize(z_mean, z_std)
                #print("z.shape",z.shape)
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

class WRN_caltech_actual(nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(WRN_caltech_actual, self).__init__()

        self.widen_factor = args.wrn_widen_factor
        self.depth = args.wrn_depth

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.dropout = args.dropout
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.nChannels = [args.wrn_embedding_size, 16 * self.widen_factor, 32 * self.widen_factor,512]
                          #64 * self.widen_factor]

        #print("total nchannekls are",self.nChannels,self)
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
        #print("total nchannekls are",self.nChannels,self.num_block_layers, self.variational,self.num_samples,self.latent_dim,self.batch_norm )
        #make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],batch_norm=True)
        model=models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
            # Add on classifier
            model.classifier[6] = nn.Sequential(nn.Linear(4096,num_classes ), nn.ReLU(), nn.Dropout(0.2),nn.Linear(num_classes, num_classes), nn.LogSoftmax(dim=1))
        self.avgpool = model.avgpool
        self.encoder =nn.Sequential(*[model.features[i] for i in range(31)])#make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],batch_norm=True)

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,self.num_colors)
        if self.variational:
            model.classifier[6] = nn.Sequential(nn.Linear(4096,self.latent_dim))
            self.latent_mu =  nn.Sequential(*[model.classifier[i] for i in range(7)])#nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels*7*7,self.latent_dim, bias=False)#self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels32768
            self.latent_std = nn.Sequential(*[model.classifier[i] for i in range(7)])#nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels*7*7,self.latent_dim, bias=False)#self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,32768,8192
            self.latent_feat_out = self.latent_dim
        else:
            self.latent_feat_out = self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels
            self.latent_dim = self.latent_feat_out
            #print(self.latent_dim)

        if self.joint:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

            if self.variational:
                self.latent_decoder = nn.Linear(self.latent_feat_out,32768)
                #self.enc_spatial_dim_x * self.enc_spatial_dim_y *self.enc_channels, bias=False)

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
        else:

            if args.train_var:
                print("ne")
                self.classifier =  nn.Sequential( nn.Linear(512, 256))
                                                            # nn.Dropout(0.2), 
                                                            # nn.Linear(512, 256), 
                                                            # nn.ReLU(), 
                                                            # nn.Dropout(0.2),
                                                            # nn.Linear(256, 256), 
                                                            # nn.LogSoftmax(dim=1))
                
            else:
                self.classifier =  nn.Sequential(*[model.classifier[i] for i in range(7)])
                
            # nn.Sequential(
            #         nn.Linear(25088, 4096),
            #         nn.ReLU(), 
            #         nn.Dropout(0.2), 
            #         nn.Linear(4096, 4096),
            #         nn.ReLU(), 
            #         nn.Dropout(0.2), 
            #         nn.Linear(4096, 256), 
            #         nn.ReLU(), 
            #         nn.Dropout(0.2),
            #         nn.Linear(256, 256), 
            #         nn.LogSoftmax(dim=1))
            # #nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

    def encode(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        #print("shape of encoder is",x.shape)#25088
        if self.variational:
            x = x.view(x.size(0), -1)
            z_mean = self.latent_mu(x)
            z_std = self.latent_std(x)
            return z_mean, z_std
        else:
            return x

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        if self.variational:
            z = self.latent_decoder(z)
            z = z.view(z.size(0),512,8,8 )#self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        #print("z.size is",z.shape)
        x = self.decoder(z)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        if self.variational:
            z_mean, z_std = self.encode(x)
            if self.joint:
                output_samples = torch.zeros(self.num_samples, x.size(0), self.num_colors, self.patch_size,
                                             self.patch_size).to(self.device)
                classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            else:
                output_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            for i in range(self.num_samples):
                z = self.reparameterize(z_mean, z_std)
                #print("z.shape",z.shape)
                if self.joint:
                    output_samples[i] = self.decode(z)
                    classification_samples[i] = self.classifier(z)
                else:
                    #print("shape of avg is",z.shape)
                    #avg=self.avgpool(z)
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
                #print("coming to the output")
                output = self.classifier(x.view(x.size(0), -1))
            return output




# args = arguments_yaml.get_args()
# with open(args.work_path) as f:
#     config = yaml.load(f)
#     # convert to dict
# args = EasyDict(config)
# tests=WRN_caltech_actual('cuda',256, 3, args)
# x = torch.randn(2, 3, 224, 224)
# tests(x)
  #print(make_vgg())
        # nn.Sequential(OrderedDict([
        #     ('encoder_conv1', nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, dropout=self.dropout)),
        #     ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, stride=2,
        #                                        dropout=self.dropout)),
        #     ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, stride=2,
        #                                        dropout=self.dropout)),
        #     ('encoder_bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.batch_norm)),
        #     ('encoder_act1', nn.ReLU(inplace=True))
        # ]))


# classifier->512->avgpool->25088->(classifier with input as 25088-4096-4096)->256 - Method 1
# classifier->512->avgpool->25088->(classifier with input as 25088)->Zmean,Zstd ->Reparmetrizatrion tric->512->Classifier- Method 2
# classifier->512->avgpool->25088->(classifier with input as 25088)->Zmean,Zstd ->Reparmetrizatrion tric->512->Classifier +Reconstruction Error- Method 3




class WRN_CIFAR100_actual(nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(WRN_CIFAR100_actual, self).__init__()
        self.widen_factor = args.wrn_widen_factor
        self.depth = args.wrn_depth
        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.dropout = args.dropout
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.nChannels = [args.wrn_embedding_size, 16 * self.widen_factor, 32 * self.widen_factor,512]
                          #64 * self.widen_factor]

        print("total nchannekls are",self.nChannels,self)
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
        print("total nchannekls are",self.nChannels,self.num_block_layers, self.variational,self.num_samples,self.latent_dim,self.batch_norm )
        #make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],batch_norm=True)

        self.encoder = make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],batch_norm=True)
        
        # nn.Sequential(OrderedDict([
        #     ('encoder_conv1', nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, dropout=self.dropout)),
        #     ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, stride=2,
        #                                        dropout=self.dropout)),
        #     ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
        #                                        WRNBasicBlock, batchnorm=self.batch_norm, stride=2,
        #                                        dropout=self.dropout)),
        #     ('encoder_bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.batch_norm)),
        #     ('encoder_act1', nn.ReLU(inplace=True))
        # ]))

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,
                                                                                          self.num_colors)
        if self.variational:
            self.latent_mu = nn.Linear(512,self.latent_dim, bias=False)#self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels32768
            self.latent_std = nn.Linear(512,self.latent_dim, bias=False)#self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,32768
            self.latent_feat_out = self.latent_dim
        else:
            self.latent_feat_out = self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels
            self.latent_dim = self.latent_feat_out
            print(self.latent_dim)

        if self.joint:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

            if self.variational:
                self.latent_decoder = nn.Linear(self.latent_feat_out,32768)
                #self.enc_spatial_dim_x * self.enc_spatial_dim_y *self.enc_channels, bias=False)

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
        else:
            self.classifier = nn.Sequential(nn.Linear(self.latent_feat_out, num_classes, bias=False))

    def encode(self, x):
        x = self.encoder(x)
        classifier_z = self.avgpool(x)
        #print("shape of encoder is",x.shape)
        if self.variational:
            x = x.view(x.size(0), -1)
            z_mean = self.latent_mu(x)
            z_std = self.latent_std(x)
            return z_mean, z_std,classifier_z
        else:
            return x

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        if self.variational:
            z = self.latent_decoder(z)
            z = z.view(z.size(0),512,8,8 )#self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        #print("z.size is",z.shape)
        x = self.decoder(z)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        if self.variational:
            z_mean, z_std,classifier_z = self.encode(x)
            if self.joint:
                output_samples = torch.zeros(self.num_samples, x.size(0), self.num_colors, self.patch_size,
                                             self.patch_size).to(self.device)
                classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            else:
                output_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            for i in range(self.num_samples):
                z = self.reparameterize(z_mean, z_std)
                #print("z.shape",z.shape)
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