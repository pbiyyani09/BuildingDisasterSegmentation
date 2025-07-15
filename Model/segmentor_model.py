import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Activation functions from TEED
@torch.jit.script
def smish(input):
    """
    Applies the smish function element-wise:
    smish(x) = x * tanh(log(1 + sigmoid(x)))
    """
    return input * torch.tanh(torch.log(1 + torch.sigmoid(input)))

class Smish(nn.Module):
    """
    Smish activation module
    """
    def __init__(self):
        super(Smish, self).__init__()

    def forward(self, input):
        return smish(input)

def apply_kaiming_init(m):
    """
    Applies Kaiming (He) initialization to Conv layers and standard
    initialization to BatchNorm layers.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # The 'relu' nonlinearity is appropriate for both ReLU and Smish
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Fusion layers adapted from TEED
class CoFusion(nn.Module):
    """
    Fusion layer from TEED to better combine features
    """
    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1)
        self.smish = Smish()
        self.norm_layer1 = nn.GroupNorm(4, 32)

    def forward(self, x):
        attn = self.smish(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        # We need to ensure the output channels match out_ch for the subsequent ConvBlock
        # The original implementation summed up the channels, resulting in a single channel.
        # Here we'll do an element-wise multiplication and a 1x1 conv to reduce channels.
        fused = x * attn
        return fused


class SeparableConvBlock(nn.Module):
    """
    Helper module for a depthwise separable convolution block.
    This is more efficient than a standard convolution.
    """
    def __init__(self, in_ch, out_ch):
        super(SeparableConvBlock, self).__init__()
        self.smish = Smish()
        self.conv = nn.Sequential(
            # Depthwise convolution (spatial filtering)
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            self.smish,
            # Pointwise convolution (channel mixing)
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            self.smish,
        )

    def forward(self, x):
        return self.conv(x)


class DoubleFusion(nn.Module):
    """
    Adapted DoubleFusion layer for multi-class segmentation.

    This module takes the concatenated multi-scale features and fuses them
    into the final class predictions, maintaining the original's inspiration
    of using two fusion blocks with a residual connection.
    """
    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        
        # A 1x1 convolution to create a residual connection path.
        # This matches the channel dimension of the main path.
        self.residual_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        # First fusion block. It processes the input features.
        self.fusion_block1 = SeparableConvBlock(in_ch, in_ch)

        # Second fusion block. It takes the fused features from the first
        # block and the residual path, and maps to the final output channels.
        self.fusion_block2 = SeparableConvBlock(in_ch, out_ch)

    def forward(self, x):
        # The main feature processing path
        fused_path = self.fusion_block1(x)
        
        # The residual connection path
        residual_path = self.residual_conv(x)
        
        # Add the residual to the processed features
        summed_features = fused_path + residual_path
        
        # Final fusion to get the output class predictions
        output = self.fusion_block2(summed_features)
        
        return output


# Atrous Spatial Pyramid Pooling module from DeepLab
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = self._aspp_branch(in_channels, out_channels, 1, dilations[0])
        self.aspp2 = self._aspp_branch(in_channels, out_channels, 3, dilations[1])
        self.aspp3 = self._aspp_branch(in_channels, out_channels, 3, dilations[2])
        self.aspp4 = self._aspp_branch(in_channels, out_channels, 3, dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            Smish()
        )

        self.conv1 = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.smish = Smish()
        self.dropout = nn.Dropout(0.5)

    def _aspp_branch(self, in_channels, out_channels, kernel_size, dilation):
        padding = 0 if kernel_size == 1 else dilation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            Smish()
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.smish(x))

        return x
    
class SeparableRecurrent_block(nn.Module):
    """
    Recurrent block using depthwise separable convolutions for greater efficiency.
    """
    def __init__(self, ch_out, t=2):
        super(SeparableRecurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        
        # Replaced the single standard convolution with a depthwise + pointwise block
        self.conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # The forward pass logic remains identical to the original Recurrent_block
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            
            x1 = self.conv(x + x1)
        return x1
    
class SeparableRRCNN_block(nn.Module):
    """
    Recurrent Residual CNN block utilizing the efficient SeparableRecurrent_block.
    """
    def __init__(self, ch_in, ch_out, t=2):
        super(SeparableRRCNN_block, self).__init__()
        
        # The key change is using the new separable block here
        self.RCNN = nn.Sequential(
            SeparableRecurrent_block(ch_out, t=t),
            SeparableRecurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # The forward pass logic remains identical to the original RRCNN_block
        x_init = self.Conv_1x1(x)
        x_rcnn = self.RCNN(x_init)
        return x_init + x_rcnn

# Improved conv block with Smish activation
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            Smish(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            Smish()
        )

    def forward(self, x):
        return self.conv(x)


# Dilated conv block
class DilatedConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation_rate=2):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=True),
            nn.BatchNorm2d(ch_out),
            Smish(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=True),
            nn.BatchNorm2d(ch_out),
            Smish()
        )

    def forward(self, x):
        return self.conv(x)

# TEED-inspired dense block
class DenseBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_layers=1):
        super(DenseBlock, self).__init__()
        self.dense_layers = nn.ModuleList()
        current_ch = ch_in
        for i in range(num_layers):
            self.dense_layers.append(self._make_dense_layer(current_ch, ch_out))
            current_ch += ch_out

        self.conv_1x1 = nn.Conv2d(current_ch, ch_out, kernel_size=1)
        self.smish = Smish()

    def _make_dense_layer(self, ch_in, ch_out):
         return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            Smish(),
        )

    def forward(self, x):
        features = [x]
        for layer in self.dense_layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)

        out = torch.cat(features, dim=1)
        out = self.conv_1x1(out)
        return self.smish(out)


# Improved up-sampling block
class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            Smish()
        )

    def forward(self, x):
        return self.up(x)


# Attention block for better feature fusion
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.smish = Smish()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.smish(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# The enhanced U-Net model combining ideas from TEED and DeepLab
class EnhancedUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=11, use_aux_loss=True):
        super(EnhancedUNet, self).__init__()

        # Encoder path
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder blocks
        t = 2
        self.enc1 = SeparableRRCNN_block(ch_in=img_ch, ch_out=64, t=t)
        self.enc2 = SeparableRRCNN_block(ch_in=64, ch_out=128, t=t)
        self.enc3 = SeparableRRCNN_block(ch_in=128, ch_out=256, t=t)
        self.enc4 = SeparableRRCNN_block(ch_in=256, ch_out=512, t=t)
        
        self.use_aux_loss = use_aux_loss # Add a flag to control it

        if self.use_aux_loss:
            self.aux_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Smish(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, output_ch, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

        # Bottleneck with ASPP
        self.bottleneck = ASPP(in_channels=512, out_channels=1024)

        # Decoder path
        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.up_conv5 = SeparableRRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.up_conv4 = SeparableRRCNN_block(ch_in=512, ch_out=256, t=t)

        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = SeparableRRCNN_block(ch_in=256, ch_out=128, t=t)

        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.up_conv2 = SeparableRRCNN_block(ch_in=128, ch_out=64, t=t)

        # Corrected final output layers
        # Create separate 1x1 convs for each decoder scale
        self.out_conv_d5 = nn.Conv2d(512, output_ch, kernel_size=1)
        self.out_conv_d4 = nn.Conv2d(256, output_ch, kernel_size=1)
        self.out_conv_d3 = nn.Conv2d(128, output_ch, kernel_size=1)
        self.out_conv_d2 = nn.Conv2d(64, output_ch, kernel_size=1)

        # Corrected final fusion layer
        # Input channels will be 4 * output_ch from the four concatenated scales
        self.final_fusion = DoubleFusion(in_ch=4 * output_ch, out_ch=output_ch)

        # Weight initialization
        print("Initializing network with Kaiming (He) initialization.")
        self.apply(apply_kaiming_init)

    def forward(self, x):
        # Encoding path
        x1 = self.enc1(x)

        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)

        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)

        x4 = self.maxpool(x3)
        x4 = self.enc4(x4)

        x5 = self.maxpool(x4)
        x5 = self.bottleneck(x5)

        # Decoding + attention path
        d5 = self.up5(x5)
        x4_att = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3_att = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1_att = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.up_conv2(d2)

        # Multi-scale output fusion (Corrected)
        out5 = F.interpolate(self.out_conv_d5(d5), size=x.shape[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(self.out_conv_d4(d4), size=x.shape[2:], mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.out_conv_d3(d3), size=x.shape[2:], mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.out_conv_d2(d2), size=x.shape[2:], mode='bilinear', align_corners=True)

        # Final fusion of multi-scale outputs
        final_out = torch.cat([out2, out3, out4, out5], dim=1)
        final_out = self.final_fusion(final_out)

        if self.training and self.use_aux_loss:
            # Pass the output of decoder block 3 (d3) to the aux head
            aux_out = self.aux_head(d3)
            return final_out, aux_out # Return both outputs during training
        else:
            return final_out # Return only the main output during inference

# Usage example
if __name__ == '__main__':
    batch_size = 2
    img_height = 256
    img_width = 256
    num_classes = 11

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model with 3 input channels and 11 output classes
    model = EnhancedUNet(img_ch=3, output_ch=num_classes, use_aux_loss=True).to(device)

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, 3, img_height, img_width).to(device)
    
    # Get model output
    output, aux_output = model(input_tensor)

    print(f"Dataset has {num_classes} classes.")
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Aux output shape: {aux_output.shape}")
    # To get the final class prediction for each pixel
    predictions = torch.argmax(output, dim=1)
    print(f"Prediction map shape: {predictions.shape}")
    aux_predictions = torch.argmax(aux_output, dim=1)
    print(f"Aux prediction map shape: {aux_predictions.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")