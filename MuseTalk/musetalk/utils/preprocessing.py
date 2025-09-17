import sys
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from musetalk.utils.face_detection import FaceAlignment, LandmarksType
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize the mmpose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# Initialize the face detection model
device_str = "cuda" if torch.cuda.is_available() else "cpu"  # Convert to string for FaceAlignment
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device_str)

# Marker if the bbox is not sufficient
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    logger.info('Reading images...')
    for img_path in tqdm(img_list, desc="Reading images"):
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"Failed to read image: {img_path}")
            continue
        frames.append(frame)
    return frames

def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        logger.info(f'Getting key landmarks and face bounding boxes with bbox_shift: {upperbondrange}')
    else:
        logger.info('Getting key landmarks and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches, desc="Processing batches"):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # Get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # Adjust the bounding box refer to landmark
        for j, f in enumerate(bbox):
            if f is None:  # No face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord = face_land_mark[29]
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]

    logger.info(f"Total frame: [{len(frames)}] Manually adjust range: [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ], current value: {upperbondrange}")
    return f"Total frame: [{len(frames)}] Manually adjust range: [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ], current value: {upperbondrange}"

def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Get landmarks and bounding boxes for a list of images.

    Parameters:
    img_list: List of image paths.
    upperbondrange: Shift value for bounding box adjustment.

    Returns:
    Tuple containing lists of coordinates and frames.
    """
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        logger.info(f'Getting key landmarks and face bounding boxes with bbox_shift: {upperbondrange}')
    else:
        logger.info('Getting key landmarks and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches, desc="Processing batches"):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # Get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # Adjust the bounding box refer to landmark
        for j, f in enumerate(bbox):
            if f is None:  # No face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord = face_land_mark[29]
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
            
            f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark
            
            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # If the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w, h = f[2] - f[0], f[3] - f[1]
                logger.warning(f"Error bbox: {f}")
            else:
                coords_list += [f_landmark]
    
    logger.info("********************************************bbox_shift parameter adjustment**********************************************************")
    logger.info(f"Total frame: [{len(frames)}] Manually adjust range: [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ], current value: {upperbondrange}")
    logger.info("*************************************************************************************************************************************")
    return coords_list, frames

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png", "./results/lyria/00001.png", "./results/lyria/00002.png", "./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        logger.info(f'Cropped shape: {crop_frame.shape}')
        
    logger.info(f"Coordinates list: {coords_list}")