import torch.nn as nn
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(self, im_size=572, channels=[3, 64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()

        self.channels = channels
        
        # Encoder (Downsampling)
        self.enc_conv0a = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.enc_conv0b = nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1)
        
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv1a = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.enc_conv1b = nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2a = nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1)
        self.enc_conv2b = nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3a = nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1)
        self.enc_conv3b = nn.Conv2d(channels[4], channels[4], kernel_size=3, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck_conv_a = nn.Conv2d(channels[4], channels[5], kernel_size=3, padding=1)
        self.bottleneck_conv_b = nn.Conv2d(channels[5], channels[5], kernel_size=3, padding=1)

        # Decoder (Upsampling)
        self.up_conv3 = nn.ConvTranspose2d(channels[5], channels[4], kernel_size=2, stride=2)
        self.dec_conv3a = nn.Conv2d(channels[4] * 2, channels[4], kernel_size=3, padding=1)
        self.dec_conv3b = nn.Conv2d(channels[4], channels[4], kernel_size=3, padding=1)

        self.up_conv2 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.dec_conv2a = nn.Conv2d(channels[3] * 2, channels[3], kernel_size=3, padding=1)
        self.dec_conv2b = nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1)

        self.up_conv1 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec_conv1a = nn.Conv2d(channels[2] * 2, channels[2], kernel_size=3, padding=1)
        self.dec_conv1b = nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1)

        self.up_conv0 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec_conv0a = nn.Conv2d(channels[1] * 2, channels[1], kernel_size=3, padding=1)
        self.dec_conv0b = nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(channels[1], 1, kernel_size=1)

        # Optional: Batch Normalization
        self.bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels[1:]])

    def forward(self, x):
        # Encoding
        x0 = F.relu(self.enc_conv0a(x))
        x0 = F.relu(self.enc_conv0b(x0))
        x1 = self.pool0(x0)

        x1 = F.relu(self.enc_conv1a(x1))
        x1 = F.relu(self.enc_conv1b(x1))
        x1 = F.dropout(x1, p=0.4, training=self.training)
        x2 = self.pool1(x1)

        x2 = F.relu(self.enc_conv2a(x2))
        x2 = F.relu(self.enc_conv2b(x2))
        x2 = F.dropout(x2, p=0.4, training=self.training)
        x3 = self.pool2(x2)

        x3 = F.relu(self.enc_conv3a(x3))
        x3 = F.relu(self.enc_conv3b(x3))
        x3 = F.dropout(x3, p=0.4, training=self.training)
        x4 = self.pool3(x3)

        # Bottleneck
        x4 = F.relu(self.bottleneck_conv_a(x4))
        x4 = F.relu(self.bottleneck_conv_b(x4))
        x4 = F.dropout(x4, p=0.4, training=self.training)

        # Decoding
        x3_up = self.up_conv3(x4)
        x3_up = torch.cat((x3_up, x3), dim=1)
        x3_up = F.relu(self.dec_conv3a(x3_up))
        x3_up = F.relu(self.dec_conv3b(x3_up))
        x3_up = F.dropout(x3_up, p=0.2, training=self.training)

        x2_up = self.up_conv2(x3_up)
        x2_up = torch.cat((x2_up, x2), dim=1)
        x2_up = F.relu(self.dec_conv2a(x2_up))
        x2_up = F.relu(self.dec_conv2b(x2_up))
        x2_up = F.dropout(x2_up, p=0.2, training=self.training)

        x1_up = self.up_conv1(x2_up)
        x1_up = torch.cat((x1_up, x1), dim=1)
        x1_up = F.relu(self.dec_conv1a(x1_up))
        x1_up = F.relu(self.dec_conv1b(x1_up))
        x1_up = F.dropout(x1_up, p=0.2, training=self.training)

        x0_up = self.up_conv0(x1_up)
        x0_up = torch.cat((x0_up, x0), dim=1)
        x0_up = F.relu(self.dec_conv0a(x0_up))
        x0_up = F.relu(self.dec_conv0b(x0_up))

        # Final convolution
        y = self.final_conv(x0_up)

        return y



class UNet_orig(nn.Module):
    #implementation to match the original paper
    
    def __init__(self, im_size = 572, channels = [3, 64, 128, 256, 512, 1024]):
        super().__init__()
        
        self.channels = channels
        c = channels #rename to write less

        # encoder (downsampling)
        self.enc_conv0a = nn.Conv2d(c[0], c[1], 3, padding=1) #im_size -> im_size - 2
        self.enc_conv0b = nn.Conv2d(c[1], c[1], 3, padding=1) #im_size - 2 -> im_size - 4
        
        self.pool0 = nn.MaxPool2d(2, 2)  # im_size - 4 -> floor(im_size/2) - 2
    
        self.enc_conv1a = nn.Conv2d(c[1], c[2], 3, padding=1) #floor(im_size/2) - 2 -> floor(im_size/2) - 4
        self.enc_conv1b = nn.Conv2d(c[2], c[2], 3, padding=1) #floor(im_size/2) - 4 -> floor(im_size/2) - 6
        
        self.pool1 = nn.MaxPool2d(2, 2)  # floor(im_size/2) - 6 -> floor(im_size/4) - 3
        
        self.enc_conv2a = nn.Conv2d(c[2], c[3], 3, padding=1) # floor(im_size/4) - 3 -> floor(im_size/4) - 5
        self.enc_conv2b = nn.Conv2d(c[3], c[3], 3, padding=1) # floor(im_size/4) - 5 -> floor(im_size/4) - 7
        
        self.pool2 = nn.MaxPool2d(2, 2) # floor(im_size/4) - 7 -> floor(im_size/8) - 3
        
        self.enc_conv3a = nn.Conv2d(c[3], c[4], 3, padding=1) # floor(im_size/8) - 3 -> floor(im_size/8) - 5
        self.enc_conv3b = nn.Conv2d(c[4], c[4], 3, padding=1) # floor(im_size/8) - 5 -> floor(im_size/8) - 7
        
        self.pool3 = nn.MaxPool2d(2, 2)  # floor(im_size/8) - 7 -> floor(im_size/16) - 3
        
        self.bottleneck_conv_a = nn.Conv2d(c[4], c[5], 3, padding=1)
        self.bottleneck_conv_b = nn.Conv2d(c[5], c[5], 3, padding=1)

        # decoder (upsampling)
        self.up_conv3 = nn.ConvTranspose2d(c[5], c[4], kernel_size = 2, stride = 2)
        
        self.dec_conv3a = nn.ConvTranspose2d(2*c[4], c[4], 3, padding = 1)
        self.dec_conv3b = nn.ConvTranspose2d(c[4], c[4], 3, padding = 1)
        
        self.up_conv2 = nn.ConvTranspose2d(c[4], c[3], kernel_size = 2, stride = 2)
        
        self.dec_conv2a = nn.ConvTranspose2d(2*c[3], c[3], 3, padding = 1)
        self.dec_conv2b = nn.ConvTranspose2d(c[3], c[3], 3, padding = 1)
        
        self.up_conv1 = nn.ConvTranspose2d(c[3], c[2], kernel_size = 2, stride = 2)
        
        self.dec_conv1a = nn.ConvTranspose2d(2*c[2], c[2], 3, padding = 1)
        self.dec_conv1b = nn.ConvTranspose2d(c[2], c[2], 3, padding = 1)
        
        self.up_conv0 = nn.ConvTranspose2d(c[2], c[1], kernel_size = 2, stride = 2)
        
        self.dec_conv0a = nn.ConvTranspose2d(2*c[1], c[1], 3, padding = 1)
        self.dec_conv0b = nn.ConvTranspose2d(c[1], c[1], 3, padding = 1)
        
        self.final_conv = nn.ConvTranspose2d(c[1], 1, 1) #1 by 1 conv
        
        self.dropout1 = nn.Dropout(p = 0.4)
        self.dropout2 = nn.Dropout(p = 0.4)
        self.dropout3 = nn.Dropout(p = 0.4)
        self.dropout4 = nn.Dropout(p = 0.4)
        
        self.dropout1_up = nn.Dropout(p = 0.2)
        self.dropout2_up = nn.Dropout(p = 0.2)
        self.dropout3_up = nn.Dropout(p = 0.2)
        
        self.bn0 = nn.BatchNorm2d(c[1])
        self.bn1 = nn.BatchNorm2d(c[2])
        self.bn2 = nn.BatchNorm2d(c[3])
        self.bn3 = nn.BatchNorm2d(c[4])
        self.bn4 = nn.BatchNorm2d(c[5])
        
    def forward(self, x):
        #encoding
        x0 = x
        x0 = F.relu(self.enc_conv0a(x0))
        x0 = F.relu(self.bn0(self.enc_conv0b(x0)))
        
        #print(f'x0.shape: {x0.shape}')
        
        x1 = self.pool0(x0)
        x1 = F.relu(self.enc_conv1a(x1))
        x1 = F.relu(self.bn1(self.enc_conv1b(x1)))
        x1 = self.dropout1(x1)
        
        #print(f'x1.shape: {x1.shape}')
        
        x2 = self.pool1(x1)
        x2 = F.relu(self.enc_conv2a(x2))
        x2 = F.relu(self.bn2(self.enc_conv2b(x2)))
        x2 = self.dropout2(x2)
        
        #print(f'x2.shape: {x2.shape}')
        
        x3 = self.pool2(x2)
        x3 = F.relu(self.enc_conv3a(x3))
        x3 = F.relu(self.bn3(self.enc_conv3b(x3)))
        x3 = self.dropout3(x3)
        
        #print(f'x3.shape: {x3.shape}')
        
        #bottleneck level
        x4 = self.pool3(x3)
        x4 = F.relu(self.bottleneck_conv_a(x4))
        x4 = F.relu(self.bn4(self.bottleneck_conv_b(x4)))
        x4 = self.dropout4(x4)
        
        #print(f'x4.shape: {x4.shape}')
        
        
        #decoding
        #level 3
        x3_up = self.up_conv3(x4)
        H, W = x3_up.shape[-2:]
        H_, W_ = x3.shape[-2:]
        diff_H, diff_W = int(H_ - H), int(W_ - W)
        skip_conn_3 = x3#[:, :, diff_H//2:-diff_H//2, diff_W//2:-diff_W//2]
        
        #print(f'H = {H}, H_ = {H_}, W = {W}, W_ = {W_}')
        #print(skip_conn_3.shape)
        #print(x3_up.shape)
        
        x3_up = torch.cat((skip_conn_3, x3_up), 1)
        x3_up = self.dec_conv3a(x3_up)
        x3_up = self.dec_conv3b(x3_up)
        x3_up = self.dropout3_up(x3_up)
        
        #print(f'x3_up.shape: {x3_up.shape}')
        
        #level 2
        x2_up = self.up_conv2(x3_up)
        H, W = x2_up.shape[-2:]
        H_, W_ = x2.shape[-2:]
        diff_H, diff_W = int(H_ - H), int(W_ - W)
        skip_conn_2 = x2#[:, :, diff_H//2:-diff_H//2, diff_W//2:-diff_W//2]
        
        x2_up = torch.cat((skip_conn_2, x2_up), 1)
        x2_up = self.dec_conv2a(x2_up)
        x2_up = self.dec_conv2b(x2_up)
        x2_up = self.dropout2_up(x2_up)
        
        #print(f'x2_up.shape: {x2_up.shape}')
        
        #level 1
        x1_up = self.up_conv1(x2_up)
        H, W = x1_up.shape[-2:]
        H_, W_ = x1.shape[-2:]
        diff_H, diff_W = int(H_ - H), int(W_ - W)
        skip_conn_1 = x1#[:, :, diff_H//2:-diff_H//2, diff_W//2:-diff_W//2]
        
        x1_up = torch.cat((skip_conn_1, x1_up), 1)
        x1_up = self.dec_conv1a(x1_up)
        x1_up = self.dec_conv1b(x1_up)
        x1_up = self.dropout1_up(x1_up)
        
        #print(f'x1_up.shape: {x1_up.shape}')
        
        #level 0
        x0_up = self.up_conv0(x1_up)
        H, W = x0_up.shape[-2:]
        H_, W_ = x0.shape[-2:]
        diff_H, diff_W = int(H_ - H), int(W_ - W)
        skip_conn_0 = x0#[:, :, diff_H//2:-diff_H//2, diff_W//2:-diff_W//2]
        
        x0_up = torch.cat((skip_conn_0, x0_up), 1)
        x0_up = self.dec_conv0a(x0_up)
        x0_up = self.dec_conv0b(x0_up)
        
        #print(f'x0_up.shape: {x0_up.shape}')
        
        #final conv
        y = self.final_conv(x0_up)
        
        #print(f'y.shape: {y.shape}')
        
        return y
    


class UNet_Legacy_1(nn.Module):
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



class UNet_Legacy_2(nn.Module):
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
    