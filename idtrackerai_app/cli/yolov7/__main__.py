import argparse
import os.path
import glob
import joblib
import logging
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
plt.set_cmap("gray")
from scipy.spatial.distance import euclidean
logging.getLogger("idtrackerai.animals_detection.yolov7").setLevel(logging.DEBUG)
logging.getLogger("idtrackerai.list_of_blobs.modifiable").setLevel(logging.DEBUG)

from confapp import conf
from idtrackerai.animals_detection.yolov7 import annotate_chunk_with_yolov7, yolov7_repo
from idtrackerai.list_of_blobs import ListOfBlobs

from .utils import validate_store


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--input", required=True, help="Folder with labels produced by YOLOv7")
    ap.add_argument("--output", default="idtrackerai", help="Output folder")
    ap.add_argument("--chunks", nargs="+", type=int, required=True)
    ap.add_argument("--session-folder", default=None, help="Folder with the results of the preprocessing (default: session_XXXXXX)")
    return ap


def main():

    ap = get_parser()
    args = ap.parse_args()
    assert len(args.chunks) == 1, "Only one chunk is supported"
    session_folder = args.session_folder
    if session_folder is None:
        session_folder = f"session_{str(args.chunks[0]).zfill(6)}"
    integrate_yolov7(store_path=args.store_path, session_folder=session_folder, n_jobs=args.n_jobs, input=args.input, output=args.output, chunks=args.chunks)


def integrate_yolov7(store_path, session_folder, n_jobs, input, output, chunks, list_of_blobs=None):
    """
    Call process_chunk and connect the blobs again

    Arguments:
        list_of_blobs (ListOfBlobs): Blobs produced in the preprocessing step.
            If not passed, they are loaded from the session_folder
    See also: process_chunk
    """

    assert os.path.exists(input), f"{input} not found (wd: {os.getcwd()}"
    assert os.path.exists(output), f"{output} not found"
    assert os.path.isdir(output), f"{output} is not a directory"
    assert len(chunks) == 1 # for now dont support more than 1 chunk
    # just call this function once per chunk in parallel
    chunk = chunks[0]

    allowed_classes={0: "fly", 1: "blurry"}
    validate_store(store_path, chunk)
    list_of_blobs, _, _, successful_frames = process_chunk(
        store_path, session_folder, chunk, input, output, allowed_classes=allowed_classes,
        list_of_blobs=list_of_blobs, return_successful_frames=True,
    )

    if getattr(conf, "INTEGRATION_FULL_OVERLAP_RECOMPUTE", False):
        # recompute overlapping, even remove the cache i.e. forget everything
        list_of_blobs.disconnect(cache=True)
        list_of_blobs.compute_overlapping_between_subsequent_frames(n_jobs=n_jobs)
    else:
        # only the overlap of the modified frames with their neighbors can change
        list_of_blobs.disconnect()
        list_of_blobs.reconnect_from_cache(skip_frames=successful_frames)
        list_of_blobs.recompute_overlapping_around_frames(successful_frames)

    video_object = np.load(os.path.join(session_folder, "video_object.npy"), allow_pickle=True).item()
    if len(list_of_blobs.blobs_in_video[video_object.episodes_start_end[0][0]]) == 0:
        logging.warning("No blobs in first frame")
    if len(list_of_blobs.blobs_in_video[video_object.episodes_start_end[-1][-1]-1]) == 0:
        logging.warning("No blobs in last frame")

    return list_of_blobs


def list_label_frames(input, chunk):
    """
    (frame_number, frame_idx) of every label produced by the AI for this chunk,
    sorted by frame number

    The labels are named {frame_number}_{chunk}-{frame_idx}.txt
    """
    labels = glob.glob(os.path.join(input, f"*_{chunk}-*.txt"))
    frames = []
    for label in labels:
        frame_number, chunk_and_idx = os.path.splitext(os.path.basename(label))[0].split("_")
        frames.append((int(frame_number), int(chunk_and_idx.split("-")[1])))
    return sorted(frames)


def process_chunk(store_path, session_folder, chunk, input, output, allowed_classes=None, logfile=None, list_of_blobs=None, return_successful_frames=False):
    """
    Improve the idtrackerai preprocessing results by collecting the output of an external AI system

    This function

       1. Lists the available labels produced by the AI system and stored under the input folder
       2. Incorporates them into an existing blobs_collection (list_of_blobs) produced by idtrackerai
       3. Produces a log of the results
           1. in the output folder in plain text format with basic stats (total # frames, # successful, # not succesful)
           2. in the integration folder in pickle format (ai.pkl) a dictionary with the succesful and unsuccessful frame numbers

    Arguments:

        * store_path (str): Path to a metadata.yaml
        * chunk (int): Chunk of the imgstore linked to the metadata to be processed
        * input (str): Folder containing labels produced by the AI system
        * output (str): Folder on which to store logs of the process
        * allowed_class (dict): If passed, only classes within the contained keys will be used, the rest will be ignored
        If None, all are used
        * list_of_blobs (ListOfBlobs): Blobs produced in the preprocessing step.
            If not passed, they are loaded from the session_folder

    Returns:
        * list_of_blobs
        * chunk (int): Passed chunk
        * sucess_rate (float): Fraction of frames processed by the AI system that meet the success criteria
        * successful_frames (list): only if return_successful_frames is True

    See annotate_chunk_with_yolov7 for details

    """

    # NOTE: this file must not be called session_XXXXXX_integration_output.txt,
    # because the pipeline redirects the output of the integration step to that file
    filename = f"session_{str(chunk).zfill(6)}_integration_stats.txt"
    if logfile is None:
        logfile = os.path.join(output, filename)

    if os.path.exists(logfile):
        os.remove(logfile)

    frames = list_label_frames(input, chunk)

    logger.debug(f"Processing {len(frames)} frames for {store_path} chunk {chunk}")
    if frames:
        list_of_blobs, successful_frames, failed_frames  = annotate_chunk_with_yolov7(
            store_path, session_folder, chunk, frames, input,
            allowed_classes=allowed_classes, exclusive=False, list_of_blobs=list_of_blobs,
        )
        processed_successfully = len(successful_frames)
        success_rate=round(processed_successfully/len(frames), 3)

    else:
        success_rate= "OK"
        processed_successfully=0
        failed_frames = []
        successful_frames=[]
        if list_of_blobs is None:
            blobs_collection = os.path.join(session_folder, "preprocessing", "blobs_collection.npy")
            assert os.path.exists(blobs_collection)
            list_of_blobs=ListOfBlobs.load(blobs_collection)

    if logfile is not None:
        with open(logfile, "a") as filehandle:
            filehandle.write(f"Processed frames: {processed_successfully}\n")
            filehandle.write(f"Total frames: {len(frames)}\n")
            filehandle.write(f"Success rate: {success_rate}\n")
    logger.info(f"YOLOv7 integration. Processed frames: {processed_successfully} Total frames: {len(frames)} Success rate: {success_rate}")

    pkl_file = os.path.join(session_folder, "integration", "ai.pkl")
    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    with open(pkl_file, "wb") as filehandle:
        pickle.dump({"success": successful_frames, "fail": failed_frames}, filehandle)

    if return_successful_frames:
        return list_of_blobs, chunk, success_rate, successful_frames
    return list_of_blobs, chunk, success_rate

if __name__ == "__main__":
     main()
