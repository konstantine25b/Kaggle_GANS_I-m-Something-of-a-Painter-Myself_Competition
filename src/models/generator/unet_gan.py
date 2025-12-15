import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=6, ngf=32):
        super(UNetGenerator, self).__init__()
        
        # Construct U-Net structure
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)  # add the innermost layer
        
        # Reduced number of intermediate layers
        # For num_downs=6, we subtract 5 (outermost + innermost + 3 reduction layers) = 1 intermediate layer
        for i in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block)
        
        # Gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)  # add the outermost layer

    def forward(self, x):
        return self.model(x)

class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

import functools

