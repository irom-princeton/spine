
from typing import Tuple, List
import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# Nerfstudio
from nerfstudio.utils.colormaps import ColormapOptions # apply_colormap


# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict

from huggingface_hub import hf_hub_download

# SAM
from torchvision.ops import box_convert
from transformers import AutoProcessor, AutoModelForCausalLM

import requests
import copy
import sam2
import open3d as o3d
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Analog to Nerfstudio's apply_colormap function.
def apply_colormap(
    image: torch.Tensor,
    colormap: str = "turbo",
    normalize: bool = True,
    eps: float = 1e-9,
):
    """
    Apply colormap to image
    """ 
    
    if normalize:
        image = image - torch.min(image)
        image = image / (torch.max(image) + eps)
        image = torch.clip(image, 0, 1)
        
    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(
        mpl.colormaps[colormap].colors,  # type: ignore
        device=image.device,
    )[image_long[..., 0]]
        
        
def compute_mIoU(
    predicted_mask: np.ndarray[bool], ground_truth_mask: np.ndarray[bool]
) -> np.ndarray:
    """
    Compute the mean Intersection over Union (mIoU) for Segmentation Tasks.
    """
    # compute the intersection
    intersection = np.logical_and(predicted_mask, ground_truth_mask)

    # compute the union
    union = np.logical_or(predicted_mask, ground_truth_mask)

    # mean intersection over union
    miou = np.sum(intersection) / np.sum(union)

    return miou

def compute_segmentation_accuracy(
    predicted_mask: np.ndarray[bool], ground_truth_mask: np.ndarray[bool]
) -> np.ndarray:
    # accuracy
    # number of correct predictions (true positive + true negative)
    correct_predictions = np.sum(predicted_mask == ground_truth_mask)

    # total number of predictions
    total_num_pred = ground_truth_mask.size

    # accuracy
    return correct_predictions / total_num_pred

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def eval_model(model, processor, image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        "cuda", torch.float16
    )
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )

    return parsed_answer


def plot_bbox(image, data):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data["bboxes"], data["labels"]):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
        )
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(
            x1,
            y1,
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    # Remove the axis ticks and labels
    ax.axis("off")

    # Show the plot
    plt.show()


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print(mask.shape)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax, label=""):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    # Annotate the label
    plt.text(
        x0, y0, label, color="white", fontsize=8, bbox=dict(facecolor="red", alpha=0.5)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            if len(np.array(box_coords).shape) > 1:
                show_box(
                    box_coords[i], plt.gca(), label="" if None else input_labels[i]
                )
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()
        
    return fig


def post_process_bbox(
    bboxes: List, labels: List, img_dim: List, max_box_dim_threshold: float = 0.9
):
    # identify the most-detailed caption/phrase for each bounding box
    bbox = {}
    for bx, lab in zip(bboxes, labels):
        bx = tuple(np.ceil(bx).astype(int).tolist())

        # remove boxes that almost span the entire image
        if (bx[2] - bx[1]) / img_dim[1] > max_box_dim_threshold and (
            bx[3] - bx[0]
        ) / img_dim[0] > max_box_dim_threshold:
            continue

        # remove erroneous labels
        if "#" in lab:
            continue

        if bx in bbox.keys():
            # retrieve the length of the caption for the bounding box
            print(lab)
            if len(lab) > bbox[bx]["caption_len"]:
                print(f"Label: {lab}")
                # update the caption
                bbox[bx]["caption"] = lab
                bbox[bx]["caption_len"] = len(bbox[bx]["caption"])
        else:
            # insert the item to the dict
            bbox[bx] = {"caption": lab, "caption_len": len(lab)}

    return bbox


# ------------------------------------------------------------------- #
# GroundingDINO
# ------------------------------------------------------------------- #


from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(
        repo_id=repo_id, filename=ckpt_config_filename
    )

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def load_image(image_pil: Image) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = image_pil.convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def load_gdino(
    ckpt_repo_id: str = "ShilongLiu/GroundingDINO",
    ckpt_filename: str = "groundingdino_swinb_cogcoor.pth",
    ckpt_config_filename: str = "GroundingDINO_SwinB.cfg.py",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Initializes GroundingDINO
    """
    # load the model
    groundingdino_model = load_model_hf(
        ckpt_repo_id, ckpt_filename, ckpt_config_filename
    ).to(device)
    
    return groundingdino_model

def get_gdino_config():
    """
    GroundingDINO config
    """
    gdino_config = {
        "BOX_TRESHOLD": 0.2,
        "TEXT_TRESHOLD": 0.2,
    }
    
    return gdino_config

# ------------------------------------------------------------------- #
# SAM2
# ------------------------------------------------------------------- #

def load_sam2(
    ckpt_name: str = "sam2.1_hiera_large.pt",
    config_path: str = "sam2.1/sam2.1_hiera_l.yaml",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Load Segment-Anything Model 2 (SAM-2)
    """
    # Load the SAM-2 model
    sam2_checkpoint = (
        f"{Path(sam2.__file__).parent.parent}/checkpoints/{ckpt_name}"
    )
    model_cfg = f"configs/{config_path}"
    sam2_model = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )

    # load the SAM-2 predictor and mask generator
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    
    return sam2_predictor, sam2_mask_generator

