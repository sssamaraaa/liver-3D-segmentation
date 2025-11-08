import torch
import torch.nn as nn


def dice_coef(pred, target, eps=1e-6):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    inter = (pred * target).sum(-1)
    denom = pred.sum(-1) + target.sum(-1)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_filters=16, deep_supervision=False):
        super().__init__()
        f = base_filters
        # encoder - downsampling
        self.inc = ConvBlock(in_ch, f)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), ConvBlock(f, f*2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), ConvBlock(f*2, f*4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), ConvBlock(f*4, f*8))
        self.down4 = nn.Sequential(nn.MaxPool3d(2), ConvBlock(f*8, f*16))
        # decoder - upsampling
        self.up3 = nn.ConvTranspose3d(f*16, f*8, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(f*16, f*8)
        self.up2 = nn.ConvTranspose3d(f*8, f*4, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(f*8, f*4)
        self.up1 = nn.ConvTranspose3d(f*4, f*2, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(f*4, f*2)
        self.up0 = nn.ConvTranspose3d(f*2, f, kernel_size=2, stride=2)
        self.conv_up0 = ConvBlock(f*2, f)
        self.outc = nn.Conv3d(f, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)        
        x2 = self.down1(x1)     
        x3 = self.down2(x2)    
        x4 = self.down3(x3)     
        x5 = self.down4(x4)     
        u3 = self.up3(x5)
        u3 = torch.cat([u3, x4], dim=1)
        u3 = self.conv_up3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.conv_up1(u1)
        u0 = self.up0(u1)
        u0 = torch.cat([u0, x1], dim=1)
        u0 = self.conv_up0(u0)
        out = self.outc(u0)
        return out

class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice = dice_coef(probs, targets)
        return self.weight_bce * bce + (1 - self.weight_bce) * (1 - dice)


