from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.video import Video
import os


class Chosen_Video(object):
    def __init__(self, processes_list=None, **kwargs):
        self.video = kwargs.get("video", Video())
        self.list_of_blobs = kwargs.get("list_of_blobs", None)
        self.list_of_fragments = kwargs.get("list_of_fragments", None)
        self.list_of_global_fragments = kwargs.get(
            "list_of_global_fragments", None
        )

        if self.video.blobs_path is not None and os.path.exists(
            self.video.blobs_path
        ):
            self.list_of_blobs = ListOfBlobs.load(self.video.blobs_path)
        else:
            self.list_of_blobs = None

        self.chosen = "Default String"
        self.processes_list = processes_list
        self.processes_to_restore = {}
        self.old_video = None

    def set_chosen_item(self, chosen_string):
        self.chosen = chosen_string

    def save(self):
        self.video.save()
        if hasattr(self, "list_of_blobs"):
            self.list_of_blobs.save(
                self.video,
                self.video.blobs_path,
            )
        if self.list_of_fragments is not None:
            self.list_of_fragments.save(self.video.fragments_path)
        if self.list_of_global_fragments is not None:
            self.list_of_global_fragments.save(
                self.video.global_fragments_path,
                self.list_of_fragments.fragments,
            )
