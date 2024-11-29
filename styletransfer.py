import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torchvision.utils import save_image
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Pretrained Feature Extractor (VGG for Perceptual Loss)
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:36])  # Use layers up to relu5_4
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

# Adaptive Instance Normalization
def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = style_feat.mean([2, 3]), style_feat.std([2, 3])
    content_mean, content_std = content_feat.mean([2, 3]), content_feat.std([2, 3])
    normalized = (content_feat - content_mean.view(size[0], size[1], 1, 1)) / content_std.view(size[0], size[1], 1, 1)
    return normalized * style_std.view(size[0], size[1], 1, 1) + style_mean.view(size[0], size[1], 1, 1)

# CLIP for Text/Style Prompt Embeddings
class CLIPStyleEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_style_prompt(self, prompt):
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

# Learned Temporal Consistency Mechanism (ConvLSTM)
class TemporalConsistencyConvLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, kernel_size=(3, 3)):
        super(TemporalConsistencyConvLSTM, self).__init__()
        self.convlstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.conv = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.convlstm(x, hidden_state)
        return self.conv(x), hidden_state

# Generator Network with AdaIN
class StyleTransferGenerator(nn.Module):
    def __init__(self, num_residual_blocks=6, style_latent_dim=512):
        super(StyleTransferGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residual_blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, x, style_features):
        x = self.encoder(x)
        x = adaptive_instance_normalization(x, style_features)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

# Optical Flow Calculation
def compute_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def warp_frame_with_flow(frame, flow):
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w))) + flow.reshape(-1, 2)
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)
    warped = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
    return warped

# Main Video Style Transfer Pipeline
def video_style_transfer(input_video, output_video, style_prompt, device='cuda'):
    # Load models
    generator = StyleTransferGenerator().to(device)
    generator.load_state_dict(torch.load("generator.pth"))  # Load pretrained generator
    clip_encoder = CLIPStyleEncoder(device)
    temporal_consistency = TemporalConsistencyConvLSTM()

    # Encode style prompt
    style_embedding = clip_encoder.encode_style_prompt(style_prompt)

    # Video processing
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    hidden_state = None

    for _ in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)
        generated_frame = generator(frame_tensor, style_embedding)

        # Apply temporal consistency
        generated_frame, hidden_state = temporal_consistency(generated_frame, hidden_state)

        generated_frame = generated_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        generated_frame = (generated_frame * 255).astype(np.uint8)
        out.write(cv2.cvtColor(generated_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    print("Video style transfer complete.")
