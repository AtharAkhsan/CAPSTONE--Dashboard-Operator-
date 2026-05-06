import torch
import torch.nn as nn
from torchvision import models


class DensityMapRegressor(nn.Module):
    """
    Model Density Map Estimation menggunakan MobileNetV2 sebagai feature extractor,
    dilanjutkan dengan 3 layer Conv2D dilated convolution untuk menangkap konteks
    objek yang tumpang tindih, dan menghasilkan density map 1 channel.
    """

    def __init__(self, pretrained=True):
        super(DensityMapRegressor, self).__init__()

        # ==============================
        # Feature Extractor: MobileNetV2
        # ==============================
        # Load pretrained MobileNetV2 dan ambil hanya bagian features-nya
        # (buang classifier akhir)
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.features = mobilenet.features  # Output: (batch, 1280, H/32, W/32)

        # ==============================
        # Dilated Convolution Layers
        # ==============================
        # 3 layer Conv2D dengan dilation=2 untuk memperluas receptive field
        # dan menangkap konteks objek yang saling tumpang tindih
        self.dilated_convs = nn.Sequential(
            # Layer 1: 1280 -> 512 channels
            nn.Conv2d(1280, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 2: 512 -> 128 channels
            nn.Conv2d(512, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3 (Final): 128 -> 1 channel (density map output)
            nn.Conv2d(128, 1, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),  # ReLU karena jumlah part tidak mungkin negatif
        )

        # ==============================
        # Upsample Layer
        # ==============================
        # Kembalikan resolusi spasial ke ukuran input (scale_factor=32)
        # karena MobileNetV2 features melakukan downscale 32x
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
            x (torch.Tensor): Input gambar dengan shape (batch, 3, H, W).

        Returns:
            torch.Tensor: Density map dengan shape (batch, 1, H, W) — sama dengan input.
        """
        # Ekstrak fitur menggunakan MobileNetV2
        x = self.features(x)

        # Proses melalui dilated convolution layers
        x = self.dilated_convs(x)

        # Upsample kembali ke resolusi input
        x = self.upsample(x)

        return x


if __name__ == '__main__':
    # ==============================
    # Quick Test
    # ==============================
    model = DensityMapRegressor(pretrained=True)
    print(model)

    # Dummy input: batch=1, 3 channel RGB, 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output min:   {output.min().item():.6f}")
    print(f"Output max:   {output.max().item():.6f}")

    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
