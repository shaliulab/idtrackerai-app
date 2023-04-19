import argparse
import logging
import os.path
from idtrackerai_app.cli import COMMANDS
from idtrackerai_app.cli.utils.submission import prepare_idtrackerai_job
logger = logging.getLogger()

def nextflow_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--command", required=True)
    return ap


def main():

    ap = nextflow_parser()
    args = ap.parse_args()
    if args.command is not None:
        if not args.command in COMMANDS:
            raise Exception("Invalid")

        
    command = args.command
    chunk = args.chunk
    store_path = args.store_path
    
    skip_saving_identification_images=False
    reconnect_blobs_from_cache=None
    skip_every_frame=1
    wait_for=[]
    submit=False
    

    if command == "preprocessing" and (reconnect_blobs_from_cache is None or reconnect_blobs_from_cache is True):
        logger.warning(f"Detected reconnect_blobs_from_cache is set to {reconnect_blobs_from_cache}. This is forbidden for command {command}, only False is allowed. I will set it to False")
        reconnect_blobs_from_cache = False

    session, cwd = prepare_idtrackerai_job(store_path, chunk, command, reconnect_blobs_from_cache, skip_saving_identification_images, skip_every_frame=skip_every_frame, wait_for=wait_for)

