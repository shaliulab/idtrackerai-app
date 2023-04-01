"""
Export scenes where the AI intervened (either successfully or not) for human supervision
Perform human supervision and validation of the annotations produced by an AI system
TODO For now only idtrackerai is featured, not YOLOv7
"""

import argparse
import os.path
import re
import os.path


import pandas as pd
import cv2
import numpy as np
import joblib
from tqdm.auto import tqdm

from idtrackerai_app.cli.utils.fragmentation import show_fragment_structure
from idtrackerai.utils.py_utils import get_spaced_colors_util
from idtrackerai.list_of_blobs import ListOfBlobs
from imgstore.interface import VideoCapture

FOLDER = "video-annotator/frames"

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True, help="Path to metadata.yaml")
    ap.add_argument("--mp4", default=False, action="store_true", help="Output format")
    ap.add_argument("--png", default=False, action="store_true", help="Output format")
    ap.add_argument("--annotate", default=False, action="store_true", help="Whether to run blob.draw() or not")
    ap.add_argument("--blob-index", default=False, action="store_true", help="Whether to write the in frame index next to the identity or not")
    ap.add_argument("--chunks", type=int, nargs="+", required=True)
    ap.add_argument("--n-jobs", type=int, default=-2)
    ap.add_argument("--window-length", type=int, default=None, help="Number of frames to show in the scene, defaults to None (framerate i.e. 1 second)")
    command_group=ap.add_mutually_exclusive_group()
    command_group.add_argument("--ai", action="store_true", default=False, help="Whether only scenes where an AI intervened should be produced")
    command_group.add_argument("--viz", action="store_true", default=False, help="Whether the whole chunk should be saved as a video file")
    return ap

def main():

    ap = get_parser()
    args = ap.parse_args()

    chunks = args.chunks
    store_path = os.path.realpath(args.store_path)

    if len(chunks) == 1:
        n_jobs=1
    else:
        n_jobs=args.n_jobs

    if args.mp4:
        extension = ".mp4"
    elif args.png:
        extension = ".png"

    if args.viz:
        ai=False
    else:
        ai=args.ai

    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(process_chunk)(store_path, chunk, extension=extension, annotate=args.annotate, blob_index=args.blob_index, ai=ai, window_length=args.window_length)
        for chunk in chunks
    )


def draw_frame(frame, blobs_in_frame, **kwargs):
    """
    Annotate the idtrackerai information onto the frame
    """

    for blob in blobs_in_frame:
        blob.draw(frame, **kwargs)

    return frame


def process_chunk(store_path, chunk, min_duration=1, extension=".png", annotate=False, ai=True, window_length=None, **kwargs):
    """
    Save scenes between two fragments where the identification network had to intervene

    half_window*2+1 frames are saved as png to the video-annotator/frames folder
    a fragment is considered significant if it at least min_duration seconds long
    a scene is considered only if the ending fragment is long enough (the next one may or may not be long enough)

    Args:

        store_path (str): Path to the imgstore recording with a
        chunk (int): Specific chunk of the imgstore recording being processed
        window_length (int): Number of frames in the scene

    """

    output_folder = os.path.join(os.path.dirname(store_path), FOLDER)
    store = VideoCapture(store_path, chunk)

    if window_length is None or window_length==0:
        framerate=store.get(5)
        window_length=framerate

    match = re.search("FlyHostel([0-9])/([0-9]*)X/([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9])/", store_path)
    flyhostel_id=int(match.group(1))
    number_of_animals=int(match.group(2))
    date_time=match.group(3)

    colors_lst = get_spaced_colors_util(number_of_animals)

    basedir = os.path.dirname(store_path)
    session_folder = os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}")

    blobs_collection = os.path.join(session_folder, "preprocessing", "blobs_collection.npy")
    if not os.path.exists(blobs_collection):
        raise FileNotFoundError(f"{blobs_collection} does not exist. Please make sure idtrackerai_preprocessing has been run")

    list_of_blobs = ListOfBlobs.load(blobs_collection)

    video_file = os.path.join(session_folder, "video_object.npy")
    video_object = np.load(video_file, allow_pickle=True).item()
    first_frame_of_chunk = video_object.episodes_start_end[0][0]
    key=f"FlyHostel{flyhostel_id}_{number_of_animals}X_{date_time}_"

    if ai:
        save_ai_intervention(
            store, list_of_blobs, chunk, output_folder,
            min_duration=min_duration, colors_lst=colors_lst,
            first_frame_number=first_frame_of_chunk,
            extension=extension,
            window_length=window_length,
            annotate=annotate,
            key=key,
            **kwargs
        )
    else:
        filename = f"{key}{str(chunk).zfill(6)}.mp4"
        chunk_center=first_frame_of_chunk+int(store._metadata["chunksize"] / 2)
        window_length=int(0.9*store._metadata["chunksize"])


        save_scene(
            store, annotation=list_of_blobs.blobs_in_video,
            scene_center=chunk_center,
            window_length=window_length,
            output_folder=None,
            output=os.path.join(output_folder, filename),
            chunk=chunk,
            extension=extension,
            annotate=annotate,
        )


def save_ai_intervention(store, list_of_blobs, chunk, output_folder, min_duration=1, extension=".png", key="", annotate=False, first_frame_number=0, **kwargs):

    structure = show_fragment_structure(chunk, min_duration=min_duration)

    for blobs_in_frame in list_of_blobs.blobs_in_video:
        if blobs_in_frame and any((blob.modified for blob in blobs_in_frame)):
            blob=blobs_in_frame[0]
            structure.append(
                ((blob.frame_number, blob.frame_number+1), None, None, None, "FOLLOWED")
            )

    first_frame = list_of_blobs.blobs_in_video[first_frame_number]

    for blob in first_frame:
        if blob.identity == 0:
            structure.append(
                ((blob.frame_number, blob.frame_number+1), None, None, None, "FOLLOWED")
            )

    os.makedirs(output_folder, exist_ok=True)
    count=0

    for scene_number, (start_end, length, significant, identity, followed) in enumerate(tqdm(structure, desc=f"Writing scenes for chunk {chunk}")):
        if followed == "FOLLOWED":
            scene_center = start_end[1]

            scene_folder=os.path.join(output_folder, f"{str(chunk).zfill(6)}_{str(scene_number+1).zfill(3)}")
            os.makedirs(scene_folder, exist_ok=True)

            if extension == ".mp4":
                if annotate:
                    suffix = "annotate"
                else:
                    suffix = "raw"
                filename = f"{key}{scene_center}_{suffix}{extension}"
                output = os.path.join(scene_folder, filename)
                scene_folder=None
            else:
                output = None

            filenames=save_scene(
                store, annotation=list_of_blobs.blobs_in_video,
                scene_center=scene_center,
                output_folder=scene_folder,
                output=output,
                chunk=chunk,
                extension=extension,
                annotate=annotate,
                first_frame_number=first_frame_number,
                **kwargs
            )
            print(f"Scene {scene_number+1}: {len(filenames)} frames")

            if filenames and extension == ".png":

                images_txt=os.path.join(scene_folder,"images.txt")
                scene_png=os.path.join(scene_folder,"scene.png")
                imgs = []
                for filename in filenames:
                    imgs.append(cv2.imread(filename))
                rows = []
                ROW_LENGTH=5
                for i in range(0, len(filenames), ROW_LENGTH):
                    row = imgs[i:(i+ROW_LENGTH)]
                    while len(row) < ROW_LENGTH:
                        row.append(np.zeros_like(row[0]))
                    rows.append(np.hstack(row))

                img=np.vstack(rows)
                cv2.imwrite(scene_png, img)
            count+=1

    print(f"{count} scenes saved")

def save_scene(store, annotation, scene_center, window_length, output_folder=None, output=None, colors_lst=None, first_frame_number=None, chunk=None, extension=None, annotate=False, blob_index=False):
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
        chunk (int): Chunk in which the scene occurs. Used just to name the output png files
        annotate (bool): Whether the contour, centroid and identity should be marked in the frame or not. Defaults to False
        blob_index (bool): Whether the blob index must be written in parenthesis next to the identity or not. Defaults to False

    Returns:
        None
    """

    progress_bar = tqdm(unit="frame")

    if colors_lst is None:
        number_of_animals=int(re.search("/([0-9]*)X/", store._basedir).group(1))
        colors_lst = get_spaced_colors_util(number_of_animals)

    assert (output_folder is None and output is not None) or (output_folder is not None and output is None)

    if output is not None:
        output_folder = os.path.dirname(output)
        filename, extension = os.path.splitext(output)
    else:
        assert extension is not None
        filename = "scene"


    if extension in [".png", ".jpg"]:
        video_writer = None
        assert first_frame_number is not None
        assert chunk is not None
    elif extension in [".mp4"]:
        video_writer = cv2.VideoWriter(
            filename=os.path.join(output_folder, f"{filename}{extension}"),
            fourcc=cv2.VideoWriter_fourcc(*"MP4V"),
            frameSize=(store.get(3), store.get(4)),
            fps=store.get(5),
            isColor=True,
        )

    start = int(scene_center - window_length // 2)
    end = int(scene_center + window_length // 2)
    frame_number = start
    store.set(1, frame_number)
    filenames=[]
    while frame_number < end:
        if video_writer is None:
            frame_idx=frame_number-first_frame_number
            filename = os.path.join(output_folder, f"{frame_number}_{chunk}-{frame_idx}.png")
            if os.path.exists(filename):
                _ = store.read()
                frame_number+=1
                continue

        frame = get_frame(store)
        if frame is None:
            break
        if annotate:
            frame = draw_frame(frame, annotation[frame_number], colors_lst=colors_lst, blob_index=blob_index)
        if video_writer is None:
            cv2.imwrite(filename, frame)
            filenames.append(filename)
        else:
            video_writer.write(frame)

        frame_number += 1
        progress_bar.update(1)

    if video_writer is not None:
        video_writer.release()

    progress_bar.close()

    # to make it a gif
    # ffmpeg -y -i video.mp4 -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0  video.gif
    return filenames


def correct_scenes_main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", type=str)
    ap.add_argument("--n-jobs", type=int, default=1)
    # ap.add_argument("--chunks", type=int, nargs="+")
    # ap.add_argument("--frame-indices", type=int, nargs="+")
    # ap.add_argument("--in-frame-indices", type=int, nargs="+")
    # ap.add_argument("--identities", type=str, nargs="+")
    ap.add_argument("--human-labels", required=True, type=str, help=".csv file with columns chunk, frame_index, in_frame_index and identity")
    args=ap.parse_args()


    basedir = os.path.dirname(args.store_path)
    human_labels_file = os.path.join(basedir, args.human_labels)

    if not os.path.exists(human_labels_file):
        raise FileNotFoundError(f"{human_labels_file} not found")

    human_labels=pd.read_csv(human_labels_file)
    kwargs={f"{column}_l": human_labels[column].tolist() for column in human_labels.columns}

    return correct_scenes(args.store_path, n_jobs=args.n_jobs, **kwargs)


def correct_scenes(store_path, chunk_l, frame_index_l, in_frame_index_l, identity_l, n_jobs=1):

    assert len(chunk_l) == len(frame_index_l) == len(in_frame_index_l) == len(identity_l)

    chunks = list(set(chunk_l))

    annotations_per_chunk = {chunk: ([], [], [], []) for chunk in chunks}

    for i, chunk in enumerate(chunk_l):
        annotations_per_chunk[chunk][0].append(frame_index_l[i])
        annotations_per_chunk[chunk][1].append(in_frame_index_l[i])
        annotations_per_chunk[chunk][2].append(identity_l[i])

    cap = VideoCapture(store_path, chunks[0])

    for chunk in chunks:
        frame_indices = annotations_per_chunk[chunk][0]
        cur = cap._index._conn.cursor()
        comma_separated_fidx = ", ".join([str(e) for e in frame_indices])
        cur.execute(
            f"SELECT frame_number FROM frames WHERE chunk = {chunk} AND frame_idx IN ({comma_separated_fidx});",
        )

        annotations_per_chunk[chunk][3].extend([row[0] for row in cur.fetchall()])


    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(process_chunk_scenes)(
            store_path, chunk,
            frame_numbers=annotations_per_chunk[chunk][3],
            in_frame_index_l = annotations_per_chunk[chunk][1],
            identity_l = annotations_per_chunk[chunk][2]
        )
        for chunk in chunks
    )


def process_chunk_scenes(store_path, chunk, frame_numbers, in_frame_index_l, identity_l):
    """Modify the identity of multiple blobs across (different) chunks

    Args:
        store_path (str): Path to metadata.yaml
        chunk_l (list): Chunk where each frame occurs
        frame_numbers (list): Frame number of each frame to be corrected
        in_frame_index_l (list): Blob in frame index of the blob to be corrected
        identity_l (list): Identities to assign to each blob

        chunks, frame_numbers, in_frame_indices and identities should have equal length
        and be equal to the blobs being corrected

    Returns:
        None
    """

    basedir = os.path.dirname(store_path)
    session_folder = os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}")
    blobs_collection = os.path.join(session_folder, "preprocessing", "blobs_collection.npy")
    if not os.path.exists(blobs_collection):
        raise FileNotFoundError(f"{blobs_collection} does not exist. Please make sure idtrackerai_preprocessing has been run")


    list_of_blobs=ListOfBlobs.load(blobs_collection)
    list_of_blobs.reconnect_from_cache()

    annotations = zip(frame_numbers, in_frame_index_l, identity_l)

    for annotation in annotations:

        frame_number, in_frame_index, identity=annotation
        blobs_in_frame = list_of_blobs.blobs_in_video[frame_number]
        blob = [blob for blob in blobs_in_frame if blob.in_frame_index == in_frame_index]
        assert len(blob) == 1
        blob=blob[0]

        if isinstance(identity, str) and ";" in identity:
            new_blob_identity = int(identity.replace(";", ""))
        elif isinstance(identity, int):
            new_blob_identity = identity

        blob.update_identity(blob.final_identities[-1], new_blob_identity, blob.centroid)
        blob._is_directly_annotated=True

        most_past_blob, most_future_blob = blob.propagate_identity(
            blob.final_identities[-1],
            str(identity),
            blob.final_centroids_full_resolution
        )

    list_of_blobs.save(blobs_collection)


def get_frame(store):


    ret, frame = store.read()
    if not ret:
        return None

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame


def view_corrections_main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", type=str)

    return view_corrections(args.store_path)


def view_corrections(store_path):
    basedir = os.path.dirname(store_path)
    file=os.path.join(basedir, "idtrackerai", "concatenation-overlap.csv")

    data=pd.read_csv(file)
    corrections=data.loc[(data["ai_identity"] == 0) | (data["ai_identity_after"] == 0)]
    print(corrections)
