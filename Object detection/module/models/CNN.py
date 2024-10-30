import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size = 256):

        self.conv_1_channels = 16
        self.conv_2_channels = 32
        self.conv_3_channels = 64
        self.conv_4_channels = 128
        self.input_size = input_size

        super(CNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.conv_1_channels, kernel_size=3, padding=1)  # 3 input channels (RGB), 8 output channels
        self.bn1 = nn.BatchNorm2d(self.conv_1_channels)  # BatchNorm

        self.conv2 = nn.Conv2d(in_channels=self.conv_1_channels, out_channels=self.conv_2_channels, kernel_size=3, padding=1)  # 16 output channels
        self.bn2 = nn.BatchNorm2d(self.conv_2_channels)  # BatchNorm

        self.conv3 = nn.Conv2d(in_channels=self.conv_2_channels, out_channels=self.conv_3_channels, kernel_size=3, padding=1)  # 32 output channels
        self.bn3 = nn.BatchNorm2d(self.conv_3_channels)  # BatchNorm

        self.conv4 = nn.Conv2d(in_channels=self.conv_3_channels, out_channels=self.conv_4_channels, kernel_size=3, padding=1)  # 32 output channels
        self.bn4 = nn.BatchNorm2d(self.conv_4_channels)  # BatchNorm

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.conv_4_channels * (self.input_size/16)**2, 256)  # 32 output channels from conv3, each 4x4 in size
        self.bn_fc1 = nn.BatchNorm1d(256)  # BatchNorm
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc_label = nn.Linear(128,1)
        self.fc_position = nn.Linear(128, 4)
        # Define the max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define dropout
        self.dropout_conv = nn.Dropout(p=0.3)  # Dropout for convolutional layers
        self.dropout_fc = nn.Dropout(p=0.4)  # Dropout for fully connected layers

    def forward(self, x):
        # Apply the first convolutional layer, batch normalization, ReLU, and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Output: 16 x 128 x 128
        x = self.dropout_conv(x)  # Apply dropout

        # Apply the second convolutional layer, batch normalization, ReLU, and max pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Output: 32 x 64 x 64
        x = self.dropout_conv(x)  # Apply dropout

        # Apply the third convolutional layer, batch normalization, ReLU, and max pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Output: 64 x 32 x 32
        x = self.dropout_conv(x)  # Apply dropout

        # Apply the third convolutional layer, batch normalization, ReLU, and max pooling
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Output: 128 x 16 x 16
        x = self.dropout_conv(x)  # Apply dropout

        # Flatten the tensor to feed into the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply the first fully connected layer, batch normalization, ReLU, and dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)  # Apply dropout

        # Apply the second fully connected layer, batch normalization, ReLU, and dropout
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)  # Apply dropout

        # Apply final layer
        y = self.fc_label(x)
        position = F.sigmoid( self.fc_position(x)) * self.input_size
        

        return y, position
