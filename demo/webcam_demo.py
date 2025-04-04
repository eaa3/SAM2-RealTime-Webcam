import os
import torch
import numpy as np
from PIL import Image
import cv2
import time
import glob  

from sam2.build_sam import build_sam2_camera_predictor, build_sam2_video_predictor

from typing import Tuple

def draw_circle(image: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    """
    Draws a circle on the provided image.

    Parameters:
        image (np.ndarray): The image on which the circle will be drawn.
        center (Tuple[int, int]): (x, y) coordinates of the circle's center.
        radius (int): Radius of the circle.
        color (Tuple[int, int, int]): Color of the circle in BGR format.
        thickness (int, optional): Thickness of the circle's outline. 
                                   Use -1 for a filled circle. Defaults to 2.

    Returns:
        np.ndarray: The image with the circle drawn on it.
    """
    # breakpoint()
    cv2.circle(image, tuple(center.astype(np.int32)), radius, color, thickness)
    return image


def add_mask_overlay(frame, out_obj_ids, out_mask_logits):
    height, width = frame.shape[:2]
    # Check mask dimensions
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()    
    if mask.shape[0] == 1:
        mask = mask.squeeze(0)  # Remove the extra dimension
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    red_mask = np.zeros((height, width, 3), dtype=np.uint8)
    red_mask[mask == 1] = [0, 0, 255]  # Red color
    alpha = 0.5  # Transparency factor
    return cv2.addWeighted(frame, 1, red_mask, alpha, 0)


device = torch.device("cuda")
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for default camera, change if using a different camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, frame = cap.read()

# read the first frame
predictor.load_first_frame(frame)
if_init = True

ann_frame_idx = 0  # the frame index we interact with

ann_obj_id = (
    1  # give a unique id to each object we interact with (it can be any integers)
)

points = np.array([[320, 240]], dtype=np.float32) # center of the image

# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], dtype=np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)



# out_obj_ids, out_mask_logits = predictor.track(frame)
overlay = add_mask_overlay(frame, out_obj_ids, out_mask_logits)



cv2.imshow("Overlay", overlay)

cv2.waitKey(1)

# start tracking videos
while 1:
    ann_frame_idx += 1

    ret, frame = cap.read()

    # Track objects in the new frame
    out_obj_ids, out_mask_logits = predictor.track(frame)

    print("out_obj_ids", out_obj_ids)
    print("out_mask_logits", out_mask_logits)
    overlay = add_mask_overlay(frame, out_obj_ids, out_mask_logits)

    draw_circle(overlay, points[0], 20, (0, 255, 0), 2)  # Draw a circle at the first point

    cv2.imshow("Overlay", overlay)

    cv2.waitKey(1)




