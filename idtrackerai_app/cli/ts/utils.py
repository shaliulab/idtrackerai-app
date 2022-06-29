import stat
import shlex
import subprocess
import os.path

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


def ts_sub(script, output, gpu=False, cwd=None, append=False):
    
    if gpu:
        gpu_requirement = "-g 0"
    else:
        gpu_requirement = "-G 0"


    if append:
        operator = ">>"
    else:
        operator = ">"
   

    ts_cmd = f"bash -c \"{script} {operator} {output} 2>&1\""
    ts_prefix = f"ts -n {gpu_requirement}"
    final_cmd=f"{ts_prefix} {ts_cmd}"
    cmd_list = shlex.split(final_cmd)
    p=subprocess.Popen(cmd_list, cwd=cwd)