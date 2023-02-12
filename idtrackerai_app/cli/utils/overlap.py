from idtrackerai.list_of_blobs.overlap import compute_overlapping_between_two_subsequent_frames
from idtrackerai.list_of_blobs import ListOfBlobs


def main():


    ap = get_parser()
    args = ap.parse_args()

    if args.chunks is None
        raise NotImplementedError
    else:
        chunks = args.chunks
    
    joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(process_chunk)(
        args.store_path, chunk 
    )
        for chunk in chunks
    )


def process_chunk(store_path, chunk)

    list_of_blobs_path = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
    list_of_blobs_path_next = os.path.join(os.path.dirname(store_path), "idtrackerai", f"session_{str(chunk+1).zfill(6)}", "preprocessing", "blobs_collection.npy")


    if os.path.exists(list_of_blobs_path) and os.path.exists(list_of_blobs_path_next):
        list_of_blobs=ListOfBlobs.load(list_of_blobs_path)
        list_of_blobs_next=ListOfBlobs.load(list_of_blobs_path_next)

        compute_overlapping_between_two_subsequent_frames(list_of_blobs.blobs_in_video[-1], list_of_blobs_next.blobs_in_video[0]
