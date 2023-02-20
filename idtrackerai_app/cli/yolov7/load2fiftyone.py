import os.path
import re
import glob
import shutil
import argparse
import tempfile
import fiftyone as fo

from idtrackerai.animals_detection.yolov7.yolov7 import load_detections
from idtrackerai.utils.py_utils import download_file, list_files

FIFTYONE_DATASETS="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/fiftyone"


def parse_experiment_from_label_file(label_file):
    pattern = os.path.join("FlyHostel([0-9])", "([0-9]*)X", "([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9])")
    result = re.search(pattern, label_file)
    flyhostel_id = int(result.group(1))
    number_of_animals = int(result.group(2))
    date_time = result.group(3)
    experiment=f"FlyHostel{flyhostel_id}_{number_of_animals}X_{date_time}"
    return experiment


def load_dataset_to_fiftyone(frames_folder, labels_folder, dataset_name, class_id=None, count=None, n_jobs=1):

    """
        frames_folder (str): Path where .png files are saved for imperfect frames that have been annotated
        labels_folder (str): Path where .txt files are saved containing the annotation of the corresponding imperfect frame
        dataset_name (str): Name of an existing fiftyone dataset
        class_id (int): Identifier of the detection's class in the fiftyone dataset
        count (int): Target number of detections in the frames
    """

    if labels_folder.startswith("http://"):
        # label_files = list_files(labels_folder, "*")
        index = tempfile.mktemp(prefix="load2fiftyone", suffix=".txt")
        label_files = []
        download_file(f"{labels_folder}/index.txt", index)
        with open(index, "r") as filehandle:
            lines = filehandle.readlines()

        for line in lines:
            fname = os.path.basename(line.strip())
            if fname != "." and fname != "index.txt":
                file = os.path.join(labels_folder, fname)
                label_files.append(file)

    else:
        assert os.path.exists(labels_folder), f"{labels_folder} not found"
        label_files = sorted(glob.glob(os.path.join(labels_folder, "*")))
        label_files = [file for file in label_files if os.path.basename(file) != "index.txt"]

    try:
        dataset = fo.load_dataset(dataset_name)
    except Exception as error:
        print(error)
        raise Exception(f"Cannot load dataset {dataset_name}. Does it exist?")

    def add_to_dataset(label_file, detections=None, add_sample=True, experiment=None):

        # the key describes the frame number chunk and frame_idx of the frame
        key = os.path.splitext(os.path.basename(label_file))[0]
        image_file = os.path.join(frames_folder, key+".png")
        # experiment describes the flyhostel setup and start time that the frame comes from

        # experiment and key combined are unique and will never repeat again
        # this way the image name makes retrieving it back in the video possible (unambiguous)

        experiment = parse_experiment_from_label_file(label_file)
        image_file_in_fiftyone = os.path.join(FIFTYONE_DATASETS, dataset_name, "images", f"{experiment}_{key}.png")

        if image_file.startswith("http://"):
            print(f"Downloading {image_file}")
            download_file(image_file, image_file_in_fiftyone)
        else:
            shutil.copy(image_file, image_file_in_fiftyone)

        if add_sample:
            assert experiment is not None
            if detections is not None:
                detections=detections2fiftyone(detections)

            sample = fo.Sample(filepath=image_file_in_fiftyone)
            sample["prediction"] = fo.Detections(detections=detections)
            sample.tags.append("load2fiftyone")
            sample.tags.append(experiment)
            dataset.add_sample(sample)
            dataset.save()


    class_names={i: dataset.default_classes[i] for i in range(len(dataset.default_classes))}
    detections = load_detections(label_files, class_id=class_id, count=count, false_action=add_to_dataset, true_action=None, n_jobs=n_jobs, class_names=class_names)
    return detections


def detections2fiftyone(detections):

    fiftyone_detections=[]
    for detection in detections:
        fiftyone_detections.append(
            fo.Detection(
                bounding_box=detection.bounding_box,
                confidence=detection.conf,
                label=detection.class_name
            )
        )

    return fiftyone_detections

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="name of fiftyone dataset")
    ap.add_argument("--frames-folder", help="path to folder containing frames input to YOLOv7, typically lots of png files")
    ap.add_argument("--labels-folder", help="path to folder containing labels produced by YOLOv7, typically one .txt file for each .png in the frames folder")
    ap.add_argument("--count", type=int, help="target number of detections in each .png", default=None)
    ap.add_argument("--n-jobs", type=int, help="Number of parallel jobs", default=1)
    ap.add_argument("--class-id", type=int, help="target class id", default=None)
    return ap

def main():

    ap = get_parser()
    args = ap.parse_args()

    detections=load_dataset_to_fiftyone(
        frames_folder=args.frames_folder, labels_folder=args.labels_folder,
        dataset_name=args.dataset, class_id=args.class_id, count=args.count, n_jobs=args.n_jobs
    )
    return detections
