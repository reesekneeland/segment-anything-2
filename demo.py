print("in demo.py!")
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import time
from sam2.build_sam import build_sam2_camera_predictor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, help="size of model", default="l")
args = parser.parse_args()
size = args.size
if size == "l":
    checkpoint = "segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
elif size == "b":
    checkpoint = "segment-anything-2/checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
elif size == "s":
    checkpoint = "segment-anything-2/checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
elif size == "t":
    checkpoint = "segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
else:
    raise ValueError("Invalid model size")
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

video_path = "/datasets/polytrauma2/GX010047.MP4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1312)
output_dir = f"runs/sam2-rt_demo_{size}/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "annotated_frames/", exist_ok=True)


def draw_mask(mask, image, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3]])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape((h, w, 1))
    mask_image = mask_image * color.reshape(1, 1, -1)
    # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2RGB)
    mask_image = (mask_image * 255).astype(np.uint8)
    mask_image = cv2.addWeighted(image, 1, mask_image, 0.5, 0)
    
    return mask_image


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

boxes = [np.array([700, 0, 900, 360], dtype=np.float32),
         np.array([525, 127, 768, 415], dtype=np.float32),
         np.array([828, 153, 1136, 485], dtype=np.float32)]

if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    iter = 0
    while True:
        ret, frame = cap.read()
        if not ret or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > 1800:
            break
        width, height = frame.shape[:2][::-1]
        if not if_init:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_dir + "output.mp4", fourcc, 30.0, (width, height))
            start = time.time()
            predictor.load_first_frame(frame)
            if_init = True
            for i, box in enumerate(boxes):
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(box=box, obj_id=i, frame_idx=iter)
                frame = draw_mask((out_mask_logits[0] > 0.0).cpu().numpy(), frame, obj_id=out_obj_ids[0])
            print(f"Prompt initialization time: {(time.time() - start)*1000:.2f}ms")
        start = time.time()
        out_obj_ids, out_mask_logits = predictor.track(frame)
        print(f"Frame {iter} time: {(time.time() - start)*1000:.2f}ms")
        for i, out_obj_id in enumerate(out_obj_ids):
            frame = draw_mask((out_mask_logits[i] > 0.0).cpu().numpy(), frame, obj_id=out_obj_id)
        
        cv2.imwrite(f"{output_dir}/annotated_frames/frame_{iter}.png", frame)
        video_writer.write(frame)
        iter +=1
    video_writer.release()