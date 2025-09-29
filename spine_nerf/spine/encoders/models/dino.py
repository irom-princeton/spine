import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    # DINO weights
    dino_weights = {
        "dinov3_vits16_pretrain_lvd1689m": """
            https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNHhqY2JwaGcwMWJ1Y3Ftd2N1NGE3MDJ6IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTcxMjQxMjB9fX1dfQ__&Signature=ok3RQ8d3relGwUyZFuA2m4hnxCjuwy471BFavHkYt66W-fCG9-DAt76rhrTsmukr2Hj7RNkUWMclKCz%7ES9IXITXp9ZxC9yqGfblH6iih%7EXhjlQnyb9XH%7E2ELST491sCPNuy5rWUDn4SCmk7YyoohR0sv5Vd4uiQlH5-Hkem49AD2V5OXgRrJvGQKlryTTAWgxzbAp2qBVHfUuzJLrBbgl1IEcnQ%7EH1GMT0HpvQ%7Ec89MqZAsUO-FctkWbI3Hf-3w9w9%7EB551Nc6kZCL88J24XPIf9u40xs0MPoeBQfAWNrScmWAwWUnZB%7EHYNXav0K806mtmw0v8d7-Bc6pJ1YVWjmA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=767546346007545
        """
    }
    
    def __init__(self, name, feature_key, dino_local_repo=None):
        super().__init__()
        self.name = name
        
        if "dinov3" in name.lower():
            # DINOv3 local repo
            if dino_local_repo is None:
                dino_local_repo =  "/n/fs/minfo/utilities/frameworks/dinov3"

            # load the base model
            self.base_model = torch.hub.load(
                dino_local_repo,
                name, 
                source="local",
                weights=DinoV2Encoder.dino_weights["dinov3_vits16_pretrain_lvd1689m"],
            )
        else:
            # load DINOv2
            self.base_model = torch.hub.load("facebookresearch/dinov2", name)
             
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb
