import torch.nn as nn
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(self, im_size = 128):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # im_size -> im_size/2
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # im_size/2 -> im_size/4
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # im_size/4 -> im_size/8
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # im_size/8 -> im_size/16

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)
        
        size_here = im_size // 16

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(size_here * 2)  # im_size/16 -> im_size/8
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(size_here * 4)  # im_size/8 -> im_size/4
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(size_here * 8)  # im_size/4 -> im_size/2
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(im_size)  # im_size/2 -> im_size
        self.dec_conv3 = nn.Conv2d(64 + 128, 1, 3, padding=1)

    def forward(self, x):
        #the enc-dec model they've written doesn't match unet. I've made some arbitrary choices to bring it closer
        #   - Like Unet, the skip connections add back in the convolved, but not pooled, tensors
        #   - Unlke Unet, i don't do two convolutions in each layer of the U
        #   - The prewritten code doesn't separate the final upconvolution from the final segmentation-convolution. For this reason, I am doing simple upsamling --> concatenation --> final convolution
        
        # encoder (i've moved the poolings to the next line)
        e0 = F.relu(self.enc_conv0(x)) #e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = F.relu(self.enc_conv1(self.pool0(e0))) #self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = F.relu(self.enc_conv2(self.pool1(e1))) #self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = F.relu(self.enc_conv3(self.pool2(e2))) #self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(self.pool3(e3)))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d0 = torch.cat((d0, e3), 1) # skip connection. 1 is the channel-axis
        
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d1 = torch.cat((d1, e2), 1) # skip connection
        
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d2 = torch.cat((d2, e1), 1)
        
        #the prewritten implementation doesn't match Unet, so I've made an arbitrary choice on how to include the final skip connection
        d2_upsampled = self.upsample3(d2)
        d2_upsampled_with_skip = torch.cat((d2_upsampled, e0), 1)
        d3 = self.dec_conv3(d2_upsampled_with_skip)  # no activation
        
        return d3



class UNet2(nn.Module):
    def __init__(self):
        super().__init__()
        

        # encoder (downsampling)
        #replace pool-layers with stride = 2 in the previous convolution
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1, stride = 2) #128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1, stride = 2) #64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1, stride = 2) #32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1, stride = 2) #16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        #replace upsample layers with stride = 2 in the previous convolution, which is changed to a transposed convolution
        self.dec_conv0 = nn.ConvTranspose2d(64, 64, 3, padding=1, stride = 2, output_padding=1) #8 -> 16
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride = 2, output_padding = 1) #16 -> 32
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride = 2, output_padding = 1) #32 -> 64
        self.dec_conv3 = nn.ConvTranspose2d(64 + 64, 1, 3, padding=1, stride = 2, output_padding = 1) #64 -> 128

    def forward(self, x):
        #since the convolution and downsampling are now happening in the same layer, we can't really copy Unet as closely without completely changing Aasa/Dmitris architecture
        #i've opted to do the skip connections using the down-sampled tensors. This means the highest-level skip connection (at resolution 128) cannot be performed
        
        # encoder
        e0 = F.relu(self.enc_conv0(x)) #64
        e1 = F.relu(self.enc_conv1(e0)) #32
        e2 = F.relu(self.enc_conv2(e1)) #16
        e3 = F.relu(self.enc_conv3(e2)) #8

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(b)) #16
        d0 = torch.cat((d0, e2), 1) # skip connection. 1 is the channel-axis
        
        d1 = F.relu(self.dec_conv1(d0)) #32
        d1 = torch.cat((d1, e1), 1) # skip connection
        
        d2 = F.relu(self.dec_conv2(d1)) #64
        d2 = torch.cat((d2, e0), 1)
        
        #cannot do the final skip connection, as the resolution is only made 128 by the final convolution
        d3 = self.dec_conv3(d2)  # no activation
        
        return d3
    