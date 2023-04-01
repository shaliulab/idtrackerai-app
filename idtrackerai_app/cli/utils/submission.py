import os.path
import json
import logging
import stat
import shutil

import yaml


SETTINGS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

assert os.path.exists(SETTINGS_JSON)
from idtrackerai.constants import ANALYSIS_FOLDER


logger = logging.getLogger(__name__)



def write_shell_script(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as filehandle:
        filehandle.write("#! /bin/bash\n")
        for line in lines:
            filehandle.write(line.rstrip("\n"))
            filehandle.write("\n")
        # filehandle.write("echo 'Hello World'")
        filehandle.write("\n")
    os.chmod(path, stat.S_IRWXU)
    return os.path.exists(path)



def make_session_script(
    config_file, chunk, command, before=[], wait_for=[]
):

    chunk_pad = str(chunk).zfill(6)
    folder = "."

    session_script = os.path.join(
        folder, f"session_{chunk_pad}", f"session_{chunk_pad}_{command}.sh"
    )
    output_file = os.path.join(
        folder, f"session_{chunk_pad}", f"session_{chunk_pad}_{command}_output.txt"
    )

    cmd = f"idtrackerai terminal_mode --_session {chunk_pad}  --load  {config_file} --exec {command}"

    label = f"{chunk}_{command}"

    path = os.path.join(
        os.path.dirname(os.path.dirname(session_script)),
        os.path.basename(session_script),
    )
    write_shell_script(session_script, [*before, cmd])
    shutil.copyfile(
        session_script,
        path,
    )
    print(f"Saving {path}")
    
    return session_script, output_file, label, wait_for, command



def symlink_video(chunk_pad, extension):

    if os.path.exists(chunk_pad + extension):
        if os.path.islink(chunk_pad + extension):
            os.remove(chunk_pad + extension)
        else:
            raise Exception(
                "An actual video has been found on the analysis folder. Only softlinks should be present"
            )
    os.symlink(os.path.join("..", chunk_pad + extension), chunk_pad + extension)

def symlink_conf(experiment):
    if not os.path.exists(f"{experiment}.conf"):
        os.symlink(
            os.path.join("..", f"{experiment}.conf"), f"{experiment}.conf"
        )
    config_file = os.path.realpath(f"{experiment}.conf")
    return config_file


def save_local_settings(chunk, **kwargs):

    chunk_pad = str(chunk).zfill(6)

    with open(SETTINGS_JSON, "r") as filehandle:
        settings = json.load(filehandle)

    for kwarg in kwargs:
        key = kwarg.upper()
        if kwargs[kwarg] is not None:
            settings[key] = kwargs[kwarg]

    lines = []
    for entry in settings:
        if type(settings[entry]) is str:
            lines.append(f'{entry}="{settings[entry]}"')
        else:
            lines.append(f"{entry}={settings[entry]}")

    lines.append(f"CHUNK={chunk}")
    path = os.path.join(f"session_{chunk_pad}-local_settings.py")
    print(f"Saving {path}")

    with open(path, "w") as filehandle:
        for line in lines:
            filehandle.write(f"{line}\n")

    return path


def prepare_idtrackerai_job(store_path, chunk, command, reconnect_blobs_from_cache, skip_saving_identification_images, skip_every_frame=1, wait_for=[]):
    """ 
    Write a shell script to the filesystem that completely encapsulates one idtrackerai pipeline step for a specific chunk of a specific experiment

    The shell script can be executed simply by runing `bash script.sh`

    Arguments:

        store_path (str): Absolute path to the imgstore to be analyzed
        chunk (int): Chunk to be processed
        command (str): idtrackerai step to be executed, one of preprocessing, integration, crossings_detection_and_fragmentation or tracking
        reconnect_blobs_from_cache (bool): Whether the blob overlapping pattern should be loaded from the list of blobs (True) or recomputed from scratch (False).
            Recomputing takes longer but ensures reproducibility if there is a problem in the cached pattern
        skip_saving_identification_images (bool): Whether idtrackerai will save the identification images, which are only needed in the last two steps of the pipeline
        skip_every_frame (int): If > 1, frames in the video are skipped, which speeds up the analysis but may compromise overlapping pattern
    
    Returns:

        session: path to shell script, path to the logfile of the script, and other info produced by make_session_script
        cwd: Working directory where the shell script must be run
    """

    chunk_pad = str(chunk).zfill(6)
    metadata_path = store_path
    basedir = os.path.dirname(store_path)
    

    # make the idtrackerai folder and move to it 
    logger.info(f"Making {basedir}/{ANALYSIS_FOLDER}")
    working_dir = os.path.join(basedir, ANALYSIS_FOLDER)
    cwd = os.path.realpath(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)


    # save the local_settings.py with the right chunk info
    local_settings_path = save_local_settings(
        chunk,
        skip_every_frame=skip_every_frame,
        skip_saving_identification_images=skip_saving_identification_images,
        reconnect_blobs_from_cache=reconnect_blobs_from_cache
    )
    copy_local_settings_cmd = f"cp {local_settings_path} local_settings.py"

    # symlink the video files from the main folder to the idtrackerai folder
    with open(metadata_path, "r") as filehandle:
        extension = yaml.load(filehandle, yaml.SafeLoader)["__store"]["extension"]
    symlink_video(chunk_pad, extension)

    # symlink the config file to idtrackerai too
    date_time = os.path.basename(basedir)
    config_file = symlink_conf(date_time)

    # write shell script
    session = make_session_script(
        config_file=config_file,
        chunk=chunk,
        command=command,
        before=[copy_local_settings_cmd],
        wait_for=wait_for,
    )

    return session, cwd
