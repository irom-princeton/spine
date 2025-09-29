from enum import Enum
# from nerfstudio.field_components.field_heads import FieldHeadNames

class SPINEFieldHeadNames(Enum):
    """Possible field outputs"""
    HASHGRID = "hashgrid"
    IMG_SEMANTICS = "img_semantics"
    LANG_SEMANTICS = "lang_semantics"
    # CLIP = "clip"
    # SAM = "sam"
    # DINO = "dino"
    # VGGT = "vggt"
