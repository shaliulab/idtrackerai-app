"""
Export scenes where the AI intervened (either successfully or not) for human supervision
"""
import argparse
import os.path

import cv2
import numpy as np
import joblib

from idtrackerai_app.cli.utils.fragmentation import show_fragment_structure
from idtrackerai.utils.py_utils import get_spaced_colors_util
from idtrackerai.list_of_blobs import ListOfBlobs
from imgstore.interface import VideoCapture

folder = "../video-annotator/frames"

def get_parser():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True, help="Path to metadata.yaml")
    ap.add_argument("--chunks", type=int, nargs="+", required=True)
    return ap

def main():
    
    ap = get_parser()
    args = ap.parse_args()
    
    chunks = args.chunks
    store_path = args.store_path

    if len(chunks) == 1:
        n_jobs=1
    else:
        n_jobs=-2
    
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(process_chunk)(store_path, chunk)
        for chunk in chunks
    )
    
    
def draw_frame(frame, blobs_in_frame, **kwargs):
    
    for blob in blobs_in_frame:
        blob.draw(frame, **kwargs)

    return frame

def process_chunk(store_path, chunk):
    
    store = VideoCapture(store_path, chunk)
    fps = store.get(5)
    
    basedir = os.path.dirname(store_path)
    session_folder = os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}")
    
    blobs_collection = os.path.join(session_folder, "preprocessing", "blobs_collection.npy")
    list_of_blobs = ListOfBlobs.load(blobs_collection)

    video_file = os.path.join(session_folder, "video_object.npy")
    video_object = np.load(video_file, allow_pickle=True).item()
    
    first_frame_of_chunk = video_object.episodes_start_end[0][0]


    colors_lst = get_spaced_colors_util(6)
    
    structure = show_fragment_structure(chunk, 1)

    for start_end, length, significant, identity, followed in structure:
        if followed:
            start = int(start_end[1] - 5)
            end = int(start_end[1] + 5)
            frame_number = start
            store.set(1, frame_number)
            while frame_number < end:
                frame_idx=frame_number - first_frame_of_chunk
                filename = os.path.join(folder, f"{frame_number}_{chunk}-{frame_idx}.png")
                if os.path.exists(filename):
                    store.read()
                    frame_number+=1
                    continue

                ret, frame = store.read()
                if not ret:
                    break


                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = draw_frame(frame, list_of_blobs.blobs_in_video[frame_number], colors_lst=colors_lst)
                os.makedirs(folder, exist_ok=True)

                cv2.imwrite(filename, frame)
                frame_number += 1
