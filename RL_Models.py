import torch
import torch.nn as nn
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock
from gymnasium import spaces
import time
import timm
import einops
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import trunc_normal_
import torchvision
from torchvision import transforms
import time
from typing import Tuple

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


import sys
sys.path.append("/home/mverghese/ego_env/Franka_Kitchen_Env")
import DP_Network
from BehaviorCloning import create_image_preprocess
sys.path.remove("/home/mverghese/ego_env/Franka_Kitchen_Env")

def create_image_preprocess(size = (240,320)):
	# TODO maybe add random cropping
	resize = torchvision.transforms.Resize(size=size)
	normalize = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	return torchvision.transforms.Compose([
		resize,
		normalize
	])

def resnet10():
    """Constructs a ResNet-10 model."""
    return ResNet(BasicBlock, [1, 1, 1, 1])

def get_pretrained_resnet10():
    model = timm.create_model(
        'resnet10t.c3_in1k',
        pretrained=True,
        features_only=False,
    )
    model.fc = nn.Identity()
    print(f"ResNet-10 model loaded with num params: {sum(p.numel() for p in model.parameters())}")
    model = model.eval()
    return model

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet

def load_DinoV2(name:str) -> nn.Module:
    """
    name: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    """
    dino_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
    return dino_model

def get_dino_preprocess_transform():
	preprocess = transforms.Compose([
		# transforms.ToTensor(),
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	return preprocess

def get_image_encoder(arch = 'ResNet-10') -> Tuple[nn.Module, torchvision.transforms.Compose, int]:
    if arch == 'ResNet-10':
        return get_pretrained_resnet10(), create_image_preprocess(), 512
    elif arch == 'ResNet-18':
        return get_resnet('resnet18', weights='IMAGENET1K_V1'), create_image_preprocess(), 512
    elif arch == 'dinov2_vits14':
        return load_DinoV2('dinov2_vits14'), get_dino_preprocess_transform(), 384
    elif arch == 'dinov2_vitb14':
        return load_DinoV2('dinov2_vitb14'), get_dino_preprocess_transform(), 768
    elif arch == 'dinov2_vitl14':
        return load_DinoV2('dinov2_vitl14'), get_dino_preprocess_transform(), 1024   
    elif arch == 'dinov2_vitg14':
        return load_DinoV2('dinov2_vitg14'), get_dino_preprocess_transform(), 1536
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def load_dp_model_from_checkpoint(checkpoint_path: str, vision_feature_dim: int = 512, action_dim: int = 8, obs_dim: int = 7, obs_horizon: int = 2):
    world_encoder = DP_Network.get_resnet('resnet18')
    world_encoder = DP_Network.replace_bn_with_gn(world_encoder)
    wrist_encoder = DP_Network.get_resnet('resnet18')
    wrist_encoder = DP_Network.replace_bn_with_gn(wrist_encoder)
    noise_pred_net = DP_Network.ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=(obs_dim + 2 * vision_feature_dim)*obs_horizon
    )

    model = nn.ModuleDict({
        'world_encoder': world_encoder,
        'wrist_encoder': wrist_encoder,
        'noise_pred_net': noise_pred_net
    })

    model.load_state_dict(torch.load(checkpoint_path))
    return model
    
class PatchEmbed2(nn.Module):
    def __init__(self, embed_dim, use_norm):
        super().__init__()
        layers = [
            nn.Conv2d(3, embed_dim, kernel_size=16, stride=8),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        ]
        self.embed = nn.Sequential(*layers)

        # self.num_patch = 121  # if input image is 96x96, then num_patch = 121
        self.num_patch = 81  # if input image is 84x84, then num_patch = 81
        self.num_patch = 266
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.embed(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y  # noqa: RET504


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        assert embed_dim % num_head == 0

        self.num_head = num_head
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        """
        x: [batch, seq, embed_dim]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head).unbind(1)
        # force flash/mem-eff attention, it will raise error if flash cannot be applied
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x  # noqa: RET504

    def _ff_block(self, x):
        x = self.linear2(nn.functional.gelu(self.linear1(x)))
        return x  # noqa: RET504


class MinVit(nn.Module):
    def __init__(self, embed_style, embed_dim, embed_norm, num_head, depth):
        super().__init__()

        if embed_style == "embed1":
            raise NotImplementedError("embed1 is not tested")
            # self.patch_embed = PatchEmbed1(embed_dim)
        if embed_style == "embed2":
            self.patch_embed = PatchEmbed2(embed_dim, use_norm=embed_norm)
        else:
            raise NotImplementedError(f"Unknown embed style {embed_style}")

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patch, embed_dim))
        layers = [TransformerLayer(embed_dim, num_head, 0) for _ in range(depth)]

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = self.patch_embed.num_patch

        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.net(x)
        return self.norm(x)

class VitEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int]):
        super().__init__()
        self.obs_shape = obs_shape
        self.vit = MinVit(
            embed_style="embed2",
            embed_dim=128,
            embed_norm=0,
            num_head=4,
            depth=1,
        )

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = 128
        self.repr_dim = 128 * self.vit.num_patches

    def forward(self, obs, flatten=True) -> torch.Tensor:
        if obs.max() > 5:
            obs = obs / 255.0
        obs = obs - 0.5
        feats: torch.Tensor = self.vit.forward(obs)
        if flatten:
            # [B, D, N] -> [B, D*N]
            feats = feats.flatten(1, 2)
        return feats



def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(fn, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        full_child_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn=fn, module=child_module, name=full_child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class CombinedResnetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, DP_Checkpoint = "", freeze_backbone: bool = True):
        '''
        ResNet Feature Extractor for processing observations.
        observation_space: The space of the observations.
        DP_Checkpoint: Path to the diffusion policy checkpoint. Used if initializing the feature extractor with the encoder trained from the diffusion policy.
        '''
        super().__init__(observation_space, 512)

        if DP_Checkpoint != "":
            dp_model = load_dp_model_from_checkpoint(DP_Checkpoint)
        self.preprocess = create_image_preprocess()
        self.freeze_backbone = freeze_backbone

        extractors = {}
        
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "world_image" or key == "wrist_image":
                if DP_Checkpoint != "":
                    encoder_key = key.replace("_image", "_encoder")
                    encoder = dp_model[encoder_key]
                else:
                    # encoder = DP_Network.get_resnet('resnet18', weights="IMAGENET1K_V1")
                    encoder = get_pretrained_resnet10()
                extractors[key] = encoder
                total_concat_size += 512  # Assuming the encoder outputs 512-dim features
            else:
                extractors[key] = nn.Identity(subspace.shape[0])
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        if freeze_backbone:
            for param in self.extractors.parameters():
                param.requires_grad = False

    def forward(self, observations: dict) -> torch.Tensor:
        encoded_tensor_list = []        
        # self.extractors contain nn.Modules that do all the processing.
        if self.freeze_backbone:
            with torch.no_grad():
                for key, extractor in self.extractors.items():
                    if key == "world_image" or key == "wrist_image":
                        image = observations[key]
                        if image.shape[-1] == 3:
                            # We have been given a channels last image
                            image = image.permute(0, 3, 1, 2)  # Change to channels first
                        preprocessed_img = self.preprocess(image)
                        encoded_tensor_list.append(extractor(preprocessed_img))
                    else:
                        encoded_tensor_list.append(extractor(observations[key]))
                # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
                ret = torch.cat(encoded_tensor_list, dim=1)
        else:
            for key, extractor in self.extractors.items():
                if key == "world_image" or key == "wrist_image":
                    image = observations[key]
                    if image.shape[-1] == 3:
                        # We have been given a channels last image
                        image = image.permute(0, 3, 1, 2)  # Change to channels first
                    preprocessed_img = self.preprocess(image)
                    encoded_tensor_list.append(extractor(preprocessed_img))
                else:
                    encoded_tensor_list.append(extractor(observations[key]))
            # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
            ret = torch.cat(encoded_tensor_list, dim=1)
        return ret

def test_init_feature_extractor(checkpoint = ""):
    obs_space = spaces.Dict({
        "world_image": spaces.Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8),
        "wrist_image": spaces.Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8),
        "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    })
    feature_extractor = CombinedResnetFeatureExtractor(obs_space, DP_Checkpoint=checkpoint, freeze_backbone=True)
    assert feature_extractor is not None

    # Test the forward pass
    dummy_obs = {
        "world_image": np.random.randint(0, 256, size=(3, 240, 320), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 256, size=(3, 240, 320), dtype=np.uint8),
        "pose": np.random.uniform(-np.pi, np.pi, size=(8,)).astype(np.float32)
    }
    batch_size = 128
    # dummy_obs = {k: torch.tensor(v).unsqueeze(0).float() for k, v in dummy_obs.items()}  # Add batch dimension
    # Create a dummy batch with batch size
    dummy_obs = {k: torch.tensor(v).unsqueeze(0).float() for k, v in dummy_obs.items()}  # Add batch dimension
    dummy_obs.update({k: v.repeat(batch_size, 1, 1, 1) for k, v in dummy_obs.items() if "image" in k})  # Repeat for batch size
    dummy_obs.update({k: v.repeat(batch_size, 1) for k, v in dummy_obs.items() if "pose" in k})  # Repeat for batch size

    for k, v in dummy_obs.items():
        print(f"Key: {k}, Value shape: {v.shape}")

    # Time the forward and backwards pass
    start_time = time.time()
    features = feature_extractor(dummy_obs)
    end_time = time.time()
    print(f"Forward pass time: {end_time - start_time:.6f} seconds")

    # Time the backward pass
    start_time = time.time()
    features.backward(torch.ones_like(features))
    end_time = time.time()
    print(f"Backward pass time: {end_time - start_time:.6f} seconds")

    print(f"Extracted features shape: {features.shape}")

if __name__ == "__main__":
    # test_init_feature_extractor(checkpoint="/home/mverghese/franka_control/open_microwave_data_10/dp_model_epoch_950.pth")
    test_init_feature_extractor()
    # resnet = get_pretrained_resnet10()
    # # List the layers in the model
    # for name, layer in resnet.named_children():
    #     print(f"Layer: {name}, Type: {layer.__class__.__name__}")
    # inputs = torch.randn(1, 3, 240, 320)
    # features = resnet(inputs)
    # print(f"Extracted features shape: {[f.shape for f in features]}")
    # ViT = VitEncoder((3, 240, 320))
    # im = torch.randn(1, 3, 240, 320)

    # start_time = time.time()
    # patch_embeds = ViT(im)
    # print(f"Encoding time: {time.time() - start_time:.4f} seconds")
    # # test backward pass
    # start_time = time.time()
    # patch_embeds.backward(torch.ones_like(patch_embeds))
    # print(f"Backward pass time: {time.time() - start_time:.4f} seconds")

    # print(f"Extracted patch embeddings shape: {patch_embeds.shape}")