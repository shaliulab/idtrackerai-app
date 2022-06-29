import argparse
import shutil
import os.path
import logging

import yaml

from imgstore.constants import STORE_MD_FILENAME
from idtrackerai.constants import ANALYSIS_FOLDER

from .utils import write_shell_script, ts_sub

logger = logging.getLogger(__name__)

def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--chunk", required=True, type=int)
    ap.add_argument("--command", required=True)
    return ap
    
def main():

    ap = get_parser()
    args=ap.parse_args()
    if not args.command in ["preprocessing", "tracking", "track_video"]:
        raise Exception("Invalid")

    os.chdir(args.root_dir)

    chunk_pad=str(args.chunk).zfill(6)
    metadata_path = os.path.join(args.experiment, STORE_MD_FILENAME)
    with open(metadata_path, "r") as filehandle:
        extension = yaml.load(filehandle, yaml.SafeLoader)["__store"]["extension"]


    logger.info(f"Making {args.experiment}/idtrackerai")
    working_dir=os.path.join(args.experiment, ANALYSIS_FOLDER)
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    lines = [
        "SETTINGS_PRIORITY=1",
        f"CHUNK={args.chunk}",
        "COLOR=False",
        "READ_FORMAT=\"imgstore\"",
        "MULTI_STORE_ENABLED=False",
        "NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION=12",
        "NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS=12",
        "NUMBER_OF_JOBS_FOR_SEGMENTATION=12",
        "NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES=12",
        "RECONNECT_BLOBS_FROM_CACHE=True",
        "MAX_RATIO_OF_PRETRAINED_IMAGES=0.80",
        "THRESHOLD_EARLY_STOP_ACCUMULATION=0.80",
        "POSTPROCESS_IMPOSSIBLE_JUMPS=False",
        "DISABLE_PROTOCOL_3=True",
    ]

    with open("local_settings.py", "w") as filehandle:
        for line in lines:
            filehandle.write(f"{line}\n")

    shutil.copy("local_settings.py", os.path.join(f"session_{chunk_pad}-local_settings.py"))
    os.remove(chunk_pad+extension)
    os.symlink(os.path.join("..", chunk_pad+extension), chunk_pad+extension)
    if not os.path.exists(f"{args.experiment}.conf"):
        os.symlink(os.path.join("..", f"{args.experiment}.conf"), f"{args.experiment}.conf")
    
    config_file=os.path.join(args.root_dir, args.experiment, args.experiment) + ".conf"
    session_script=os.path.join(".", f"session_{chunk_pad}", f"session_{chunk_pad}.sh")
    output_file=os.path.join(f"session_{chunk_pad}", f"session_{chunk_pad}_output.txt")

    cmd = f"idtrackerai terminal_mode --_session {chunk_pad}  --load  {config_file} --exec {args.command}"
    logger.debug(f"Running {cmd} with ts")

    write_shell_script(session_script, [cmd])
    cwd=os.path.join(args.root_dir, args.experiment, ANALYSIS_FOLDER)
    ts_sub(session_script, output_file, gpu=args.command=="tracking", cwd=cwd)      
    os.chdir(args.root_dir)
