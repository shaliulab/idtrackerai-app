import os.path


def get_blobs_collection(root, experiment, chunk):
    path = os.path.join(root, experiment, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
    return path

