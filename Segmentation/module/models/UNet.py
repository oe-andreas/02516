import torch.nn as nn
import torch.nn.functional as F
import torch

class UNet_orig(nn.Module):
    #implementation to match the original paper
    
    def __init__(self, im_size = 572, channels = [3, 64, 128, 256, 512, 1024]):
        super().__init__()
        
        self.channels = channels
        c = channels #rename to write less
        

        # encoder (downsampling)
        self.enc_conv0a = nn.Conv2d(c[0], c[1], 3, padding=0) #im_size -> im_size - 2
        self.enc_conv0b = nn.Conv2d(c[1], c[1], 3, padding=0) #im_size - 2 -> im_size - 4
        
        self.pool0 = nn.MaxPool2d(2, 2)  # im_size - 4 -> floor(im_size/2) - 2
    
        self.enc_conv1a = nn.Conv2d(c[1], c[2], 3, padding=0) #floor(im_size/2) - 2 -> floor(im_size/2) - 4
        self.enc_conv1b = nn.Conv2d(c[2], c[2], 3, padding=0) #floor(im_size/2) - 4 -> floor(im_size/2) - 6
        
        self.pool1 = nn.MaxPool2d(2, 2)  # floor(im_size/2) - 6 -> floor(im_size/4) - 3
        
        self.enc_conv2a = nn.Conv2d(c[2], c[3], 3, padding=0) # floor(im_size/4) - 3 -> floor(im_size/4) - 5
        self.enc_conv2b = nn.Conv2d(c[3], c[3], 3, padding=0) # floor(im_size/4) - 5 -> floor(im_size/4) - 7
        
        self.pool2 = nn.MaxPool2d(2, 2) # floor(im_size/4) - 7 -> floor(im_size/8) - 3
        
        self.enc_conv3a = nn.Conv2d(c[3], c[4], 3, padding=0) # floor(im_size/8) - 3 -> floor(im_size/8) - 5
        self.enc_conv3b = nn.Conv2d(c[4], c[4], 3, padding=0) # floor(im_size/8) - 5 -> floor(im_size/8) - 7
        
        self.pool3 = nn.MaxPool2d(2, 2)  # floor(im_size/8) - 7 -> floor(im_size/16) - 3
        
        self.bottleneck_conv_a = nn.Conv2d(c[4], c[5], 3, padding=0)
        self.bottleneck_conv_b = nn.Conv2d(c[5], c[5], 3, padding=0)
)
        # decoder (upsampling)
        self.up_conv3 = nn.ConvTranspose2d(c[5], c[4], kernel_size = 2, stride = 2)
        
        self.dec_conv3a = nn.ConvTranspose2d(2*c[4], c[4], 3, padding = 0)
        self.dec_conv3b = nn.ConvTranspose2d(c[4], c[4], 3, padding = 0)
        
        self.up_conv2 = nn.ConvTranspose2d(c[4], c[3], kernel_size = 2, stride = 2)
        
        self.dec_conv2a = nn.ConvTranspose2d(2*c[3], c[3], 3, padding = 0)
        self.dec_conv2b = nn.ConvTranspose2d(c[3], c[3], 3, padding = 0)
        
        self.up_conv1 = nn.ConvTranspose2d(c[3], c[2], kernel_size = 2, stride = 2)
        
        self.dec_conv1a = nn.ConvTranspose2d(2*c[2], c[2], 3, padding = 0)
        self.dec_conv1b = nn.ConvTranspose2d(c[2], c[2], 3, padding = 0)
        
        self.up_conv0 = nn.ConvTranspose2d(c[2], c[1], kernel_size = 2, stride = 2)
        
        self.dec_conv0a = nn.ConvTranspose2d(2*c[1], c[1], 3, padding = 0)
        self.dec_conv0b = nn.ConvTranspose2d(c[1], c[1], 3, padding = 0)
        
        self.final_conv = nn.ConvTranspose2d(c[1], 1, 1, padding = 0) #1 by 1 conv
        
    def forward(self, x):
        #encoding
        x0 = x
        x0 = F.relu(self.enc_conv0a(x0))
        x0 = F.relu(self.enc_conv0b(x0))
        
        x1 = self.pool0(x0)
        x1 = F.relu(self.enc_conv1a(x1))
        x1 = F.relu(self.enc_conv1b(x1))
        
        x2 = self.pool1(x1)
        x2 = F.relu(self.enc_conv2a(x2))
        x2 = F.relu(self.enc_conv2b(x2))
        
        x3 = self.pool2(x2)
        x3 = F.relu(self.enc_conv3a(x3))
        x3 = F.relu(self.enc_conv3b(x3))
        
        
        #bottleneck level
        x4 = self.pool3(x3)
        x4 = F.relu(self.bottleneck_conv_a(x4))
        x4 = F.relu(self.bottleneck_conv_b(x4))
        
        
        #decoding
        #level 3
        x3_up = self.up_conv3(x4)
        H, W = x3_up.shape[-2:]
        H_, W_ = x3.shape[-2:]
        diff_H, diff_W = H_ - H, W_ - W
        skip_conn_3 = x3[:, :, diff_H/2:-diff_H/2, diff_W/2:-diff_W/2]
        
        x3_up = torch.cat((skip_conn_3, x3_up), 1)
        x3_up = self.dec_conv3a(x3_up)
        x3_up = self.dec_conv3b(x3_up)
        
        #level 2
        x2_up = self.up_conv2(x3_up)
        H, W = x2_up.shape[-2:]
        H_, W_ = x2.shape[-2:]
        diff_H, diff_W = H_ - H, W_ - W
        skip_conn_2 = x2[:, :, diff_H/2:-diff_H/2, diff_W/2:-diff_W/2]
        
        x2_up = torch.cat((skip_conn_2, x2_up), 1)
        x2_up = self.dec_conv2a(x2_up)
        x2_up = self.dec_conv2b(x2_up)
        
        #level 1
        x1_up = self.up_conv1(x1_up)
        H, W = x1_up.shape[-2:]
        H_, W_ = x1.shape[-2:]
        diff_H, diff_W = H_ - H, W_ - W
        skip_conn_1 = x1[:, :, diff_H/2:-diff_H/2, diff_W/2:-diff_W/2]
        
        x1_up = torch.cat((skip_conn_1, x1_up), 1)
        x1_up = self.dec_conv1a(x1_up)
        x1_up = self.dec_conv1b(x1_up)
        
        #level 0
        x0_up = self.up_conv0(x1_up)
        H, W = x0_up.shape[-2:]
        H_, W_ = x0.shape[-2:]
        diff_H, diff_W = H_ - H, W_ - W
        skip_conn_0 = x0[:, :, diff_H/2:-diff_H/2, diff_W/2:-diff_W/2]
        
        x0_up = torch.cat((skip_conn_0, x0_up), 1)
        x0_up = self.dec_conv0a(x0_up)
        x0_up = self.dec_conv0b(x0_up)
        
        #final conv
        y = self.final_conv(x0_up)
        
        return y



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
    