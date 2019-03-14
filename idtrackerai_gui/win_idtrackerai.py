import os

from PyQt5.QtWidgets import QApplication
from pyforms.controls import ControlButton
from .base_idtrackerai import BaseIdTrackerAi
from pythonvideoannotator_module_idtrackerai.idtrackerai_importer import import_idtrackerai_project
from pythonvideoannotator.__main__ import start as start_videoannotator
from pythonvideoannotator_models.models import Project


class IdTrackerAiGUI(BaseIdTrackerAi):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._editpaths = ControlButton('Edit paths', default=self.__open_videoannotator_evt, enabled=False)

        self.formset = [
            ('_video', '_session'),
            '_player',
            '=',
            ('_range', '_rangelst', '_addrange', '_multiple_range'),
            '_intensity',
            ('_area', '_togglegraph'),
            ('_nblobs', '_resreduct', ' ', '_applyroi', '_chcksegm', '_bgsub'),
            ('_polybtn', '_rectbtn', '_circlebtn', ' '),
            '_roi',
            ('_no_ids', '_pre_processing', '_savebtn', '_progress', '_editpaths')
        ]


        self._editpaths.enabled = True

    def __update_progress_evt(self, progress_count, max_count=None):
        if max_count is not None:
            self._progress.max = max_count
            self._progress.value = 0
            self._progress.show()
        elif self._progress.max == progress_count:
            self._progress.hide()
        else:
            self._progress.value = progress_count

    def __open_videoannotator_evt(self):

        video_folder = os.path.dirname(self._video.value)
        session_folder = "session_{0}".format(self._session.value)

        session_path = os.path.join(video_folder, session_folder)
        annotator_projpath = os.path.join(session_path, 'videoannotator-project')
        proj = Project()
        import_idtrackerai_project(proj, session_path, self.__update_progress_evt)
        proj.save(project_path=annotator_projpath)


        videoannotator_app = start_videoannotator(parent_win=self)
        QApplication.processEvents()
        videoannotator_app.load_project(annotator_projpath)