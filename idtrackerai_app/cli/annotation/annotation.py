"""
Export scenes where the AI intervened (either successfully or not) for human supervision
"""
"""
Perform human supervision and validation of the annotations produced by an AI system
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

FOLDER = "video-annotator/frames"

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
    """
    Annotate the idtrackerai information onto the frame
    """

    for blob in blobs_in_frame:
        blob.draw(frame, **kwargs)

    return frame


def process_chunk(store_path, chunk, min_duration=1, **kwargs):
    """
    Save scenes between two fragments where the identification network had to intervene

    half_window*2+1 frames are saved as png to the video-annotator/frames folder
    a fragment is considered significant if it at least min_duration seconds long
    a scene is considered only if the ending fragment is long enough (the next one may or may not be long enough)

    Args:

        store_path (str): Path to the imgstore recording with a
        chunk (int): Specific chunk of the imgstore recording being processed

    """

    output_folder = os.path.join(os.path.dirname(os.path.realpath(store_path)), FOLDER)

    store = VideoCapture(store_path, chunk)
    number_of_animals = int(re.search("FlyHostel[0-9]/([0-9])*X/", store_path).group(1))
    colors_lst = get_spaced_colors_util(number_of_animals)

    basedir = os.path.dirname(store_path)
    session_folder = os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}")

    blobs_collection = os.path.join(session_folder, "preprocessing", "blobs_collection.npy")
    if not os.path.exists(blobs_collection):
        raise Exception(f{"blobs_collection} does not exist. Please make sure idtrackerai_preprocessing has been run")

    list_of_blobs = ListOfBlobs.load(blobs_collection)

    video_file = os.path.join(session_folder, "video_object.npy")
    video_object = np.load(video_file, allow_pickle=True).item()
    first_frame_of_chunk = video_object.episodes_start_end[0][0]

    structure = show_fragment_structure(chunk, min_duration=min_duration)
    os.makedirs(output_folder, exist_ok=True)

    for start_end, length, significant, identity, followed in structure:
        if followed == "FOLLOWED":
            fragment_end = start_end[1]
            save_scene(store, annotation=list_of_blobs.blobs_in_video, scene_center=fragment_end, first_frame_number=first_frame_of_chunk, colors_lst=colors_lst, output_folder=output_folder)

def save_scene(store, annotation, scene_center, window_length, first_frame_number, colors_lst, output_folder):
    """
    Save the annotated frames produced by the idtracker AI in complex scene for human validation

    Args:

        store (imgstore.stores.VideoStore)
        annotation (list): AI annotation, list indexed by frame numbers,
            where each value is a list of objects with a draw method taking the frame as argument.
            The method must modify the frame in place
        scene_center (int): Frame number around which the scene revolves
        window_length (int):  Number of frames contained in the scene, half before and half after the scene_center
        first_frame_number (int): The difference between each frame's frame number and this number is the frame_idx
        colors_lst (list): List of tuples where the ith element is a 3 number tuple with the RGB code of the ith identity's color
        output_folder (str): Path to the directory where the annotated frames will be saved

    Returns
        None
    """

    start = int(scene_center - window_length // 2)
    end = int(scene_center + window_length // 2)
    frame_number = start
    store.set(1, frame_number)
    while frame_number < end:
        frame_idx=frame_number-first_frame_number
        filename = os.path.join(output_folder, f"{frame_number}_{chunk}-{frame_idx}.png")
        if os.path.exists(filename):
            _ = store.read()
            frame_number+=1
            continue

        frame = get_frame(store)
        frame = draw_frame(frame, annotation[frame_number], colors_lst=colors_lst)
        cv2.imwrite(filename, frame)
        frame_number += 1



def get_frame(store):

    ret, frame = store.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame
