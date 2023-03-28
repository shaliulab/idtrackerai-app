import os.path
import argparse
import subprocess



def parse_token(token):

    tokens = token.split(":")

    if len(tokens) == 1:

        server = None
        path = tokens[0]
    else:
        server, path = tokens

    return server, path

def transfer_file(src, dst, update=True):

    if update:
        cmd = ["rsync", "-u", src, dst]
    else:
        cmd = ["rsync", src, dst]

    p = subprocess.Popen(cmd)

    return p.communicate()


def transfer_folder(src, dst):

    cmd = ["scp", "-r", src, dst]
    print(cmd)
    p = subprocess.Popen(cmd)

    return p.communicate()
