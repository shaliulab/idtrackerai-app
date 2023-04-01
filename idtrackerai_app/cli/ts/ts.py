import argparse
import shutil
import json
import os.path
import logging

import yaml

from imgstore.constants import STORE_MD_FILENAME
from idtrackerai_app.cli.utils.submission import prepare_idtrackerai_job
from idtrackerai_app.cli.ts.utils import ts_sub
from idtrackerai_app.cli import COMMANDS


logger = logging.getLogger(__name__)


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--chunk", required=True, type=int)
    ap.add_argument("--command", required=False)
    ap.add_argument("--wait_for", type=int, nargs="+", required=False, default=[])
    ap.add_argument("--reconnect-blobs-from-cache", action="store_true", default=None)
    ap.add_argument("--submit", action="store_true", dest="submit", default=True)
    ap.add_argument("--no-submit", action="store_false", dest="submit", default=True)
    ap.add_argument("--skip-every-frame", default=1, type=int, help="If differnt from 1, only every skip-every-frameth is analyzed, and all others will have no blobs")
    return ap


def main():

    ap = get_parser()
    args = ap.parse_args()
    if args.command is not None:
        if not args.command in COMMANDS:
            raise Exception("Invalid")
        
        
    root_dir = args.root_dir
    
    chunk = args.chunk
    experiment = args.experiment
    reconnect_blobs_from_cache=args.reconnect_blobs_from_cache
    skip_every_frame=args.skip_every_frame
    submit=args.submit
    command=args.command
    wait_for=args.wait_for
    store_path=os.path.join(root_dir, experiment, STORE_MD_FILENAME)
    skip_saving_identification_images=False


    # IMPORTANT
    # reconnect_blobs_from_cache must be False in this step
    # to keep the reconnect_from_cache function from setting blobs_are_connected to True
    # when there is no cache! 

    if command == "preprocessing" and (reconnect_blobs_from_cache is None or reconnect_blobs_from_cache is True):
        logger.warning(f"Detected reconnect_blobs_from_cache is set to {reconnect_blobs_from_cache}. This is forbidden for command {command}, only False is allowed. I will set it to False")
        reconnect_blobs_from_cache = False

    push_idtrackerai_job_to_ts(root_dir, store_path, chunk, command, reconnect_blobs_from_cache, skip_saving_identification_images, skip_every_frame=skip_every_frame, wait_for=wait_for, submit=submit)


def push_idtrackerai_job_to_ts(root_dir, *args, submit=True, **kwargs):

    session, cwd = prepare_idtrackerai_job(*args, **kwargs)
    
    if submit:
        preprocessing_id = submit_to_ts(*session, wd=cwd)

    os.chdir(root_dir)


def submit_to_ts(session_script, output_file, label, wait_for, command, wd):

    print(f"Submitting job with label: {label}")

    process_id = ts_sub(
        session_script,
        output_file,
        gpu=True,
        cwd=wd,
        append=command == "tracking",
        label=label,
        wait_for=wait_for,
    )
    return process_id

