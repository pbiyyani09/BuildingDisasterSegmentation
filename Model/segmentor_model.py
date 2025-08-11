import torch
import torch.nn as nn
import torch.nn.functional as F

# Smish activation (from TEED)
@torch.jit.script
def smish(input):
    """Smooth, non-monotonic activation function"""
    return input * torch.tanh(torch.log(1 + torch.sigmoid(input)))

class Smish(nn.Module):
    def forward(self, input):
        return smish(input)

# Efficient double convolution block with separable convolutions
class EfficientConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_residual=True):
        super(EfficientConvBlock, self).__init__()
        self.use_residual = use_residual and (in_ch == out_ch)
        
        # First depthwise separable convolution
        self.dwconv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.pwconv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)
        
        # Second depthwise separable convolution
        self.dwconv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False)
        self.norm3 = nn.BatchNorm2d(out_ch)
        self.pwconv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.norm4 = nn.BatchNorm2d(out_ch)
        
        self.act = Smish()
        
    def forward(self, x):
        # First separable conv
        out = self.dwconv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.pwconv1(out)
        out = self.norm2(out)
        out = self.act(out)
        
        # Second separable conv
        identity = out
        out = self.dwconv2(out)
        out = self.norm3(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.norm4(out)
        
        # Residual connection if input/output channels match
        if self.use_residual:
            out = out + identity
            
        out = self.act(out)
        return out

# Lightweight ASPP module (memory-efficient)
class LightASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LightASPP, self).__init__()
        reduced_ch = in_ch // 4  # Reduce internal channels
        
        # Parallel dilated convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, reduced_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_ch),
            Smish()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, reduced_ch, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(reduced_ch),
            Smish()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, reduced_ch, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(reduced_ch),
            Smish()
        )
        
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, reduced_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_ch),
            Smish()
        )
        
        # Final 1x1 conv to combine features
        self.combine = nn.Sequential(
            nn.Conv2d(reduced_ch*4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            Smish()
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        b4 = self.branch4(x)
        b4 = F.interpolate(b4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine branches
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.combine(out)
        
        return out

# TEED-inspired attention gate
class EfficientAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(EfficientAttention, self).__init__()
        # Reduce internal channels
        F_int = max(F_int, 8)  # Minimum internal channels
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.act = Smish()
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Efficient upsampling block
class EfficientUpconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EfficientUpconv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            Smish()
        )
        
    def forward(self, x):
        return self.up(x)

# TEED-inspired sequential feature fusion
class SequentialFusion(nn.Module):
    def __init__(self, output_ch):
        super(SequentialFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(output_ch*2, output_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_ch),
            Smish()
        )
        
    def forward(self, *features):
        # Start with first feature
        result = features[0]
        
        # Sequentially fuse each feature
        for i in range(1, len(features)):
            result = self.conv(torch.cat([result, features[i]], dim=1))
            
        return result

# Main model: TEEDInspiredAttUNet
class TEEDInspiredAttUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=11):
        super(TEEDInspiredAttUNet, self).__init__()
        
        # Reduce base channels (TEED philosophy)
        base_ch = 8  # Starting with 32 channels instead of 64
        
        # Encoder blocks
        self.enc1 = EfficientConvBlock(img_ch, base_ch)
        self.enc2 = EfficientConvBlock(base_ch, base_ch*2)
        self.enc3 = EfficientConvBlock(base_ch*2, base_ch*4)
        self.enc4 = EfficientConvBlock(base_ch*4, base_ch*8)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck with lightweight ASPP
        self.bottleneck = LightASPP(base_ch*8, base_ch*16)
        
        # Decoder blocks with attention
        self.up5 = EfficientUpconv(base_ch*16, base_ch*8)
        self.att5 = EfficientAttention(F_g=base_ch*8, F_l=base_ch*8, F_int=base_ch*4)
        self.dec5 = EfficientConvBlock(base_ch*16, base_ch*8)
        
        self.up4 = EfficientUpconv(base_ch*8, base_ch*4)
        self.att4 = EfficientAttention(F_g=base_ch*4, F_l=base_ch*4, F_int=base_ch*2)
        self.dec4 = EfficientConvBlock(base_ch*8, base_ch*4)
        
        self.up3 = EfficientUpconv(base_ch*4, base_ch*2)
        self.att3 = EfficientAttention(F_g=base_ch*2, F_l=base_ch*2, F_int=base_ch)
        self.dec3 = EfficientConvBlock(base_ch*4, base_ch*2)
        
        self.up2 = EfficientUpconv(base_ch*2, base_ch)
        self.att2 = EfficientAttention(F_g=base_ch, F_l=base_ch, F_int=base_ch//2)
        self.dec2 = EfficientConvBlock(base_ch*2, base_ch)
        
        # Multi-scale outputs
        self.out5 = nn.Conv2d(base_ch*8, output_ch, kernel_size=1)
        self.out4 = nn.Conv2d(base_ch*4, output_ch, kernel_size=1)
        self.out3 = nn.Conv2d(base_ch*2, output_ch, kernel_size=1)
        self.out2 = nn.Conv2d(base_ch, output_ch, kernel_size=1)
        
        # Final fusion
        self.fusion = SequentialFusion(output_ch)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Input size tracking for upsampling
        input_size = x.shape[2:]
        
        # Encoding path
        e1 = self.enc1(x)
        
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        
        e3 = self.pool(e2)
        e3 = self.enc3(e3)
        
        e4 = self.pool(e3)
        e4 = self.enc4(e4)
        
        # Bottleneck
        b = self.pool(e4)
        b = self.bottleneck(b)
        
        # Decoding path with attention
        d5 = self.up5(b)
        e4 = self.att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        e3 = self.att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e2 = self.att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e1 = self.att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.dec2(d2)
        
        # Multi-scale outputs
        s5 = F.interpolate(self.out5(d5), size=input_size, mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.out4(d4), size=input_size, mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.out3(d3), size=input_size, mode='bilinear', align_corners=False)
        s2 = self.out2(d2)
        
        # Sequential fusion of multi-scale predictions
        out = self.fusion(s2, s3, s4, s5)
        
        return out

# Usage example
if __name__ == '__main__':
    batch_size = 4
    img_height = 512
    img_width = 512
    num_classes = 11

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model with 3 input channels and 11 output classes
    model = TEEDInspiredAttUNet(img_ch=3, output_ch=num_classes).to(device)

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, 3, img_height, img_width).to(device)
    
    # Get model output
    output = model(input_tensor)

    print(f"Dataset has {num_classes} classes.")
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    # To get the final class prediction for each pixel
    predictions = torch.argmax(output, dim=1)
    print(f"Prediction map shape: {predictions.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")