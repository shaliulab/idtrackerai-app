import os, subprocess, cv2, numpy as np, sys
from confapp import conf

from idtrackerai.utils.segmentation_utils import segment_frame, blob_extractor

from pyforms.controls import ControlButton
from pyforms.controls import ControlPlayer
from pyforms.controls import ControlCheckBox

from .base_idtrackerai import BaseIdTrackerAi
from .gui.grapharea_win import GraphAreaWin
from .gui.range_win import RangeWin


class IdTrackerAiGUI(BaseIdTrackerAi):

    def __init__(self, *args, **kwargs):

        self._player = ControlPlayer('Player', enabled=False, multiple_files=True)
        self._togglegraph = ControlCheckBox('Segmented blobs info', changed_event=self.__toggle_graph_evt, enabled=False)
        self._editpaths      = ControlButton('Validate trajectories', default=self.__open_videoannotator_evt, enabled=False)
        self._pre_processing = ControlButton('Track video', default=self.track_video, enabled=False)
        self._savebtn        = ControlButton('Save parameters', default=self.save_window, enabled=False)

        self._polybtn = ControlButton('Polygon', checkable=True, enabled=False, default=self.polybtn_click_evt)
        self._rectbtn = ControlButton('Rectangle', checkable=True, enabled=False)
        self._circlebtn = ControlButton('Ellipse', checkable=True, enabled=False, default=self.circlebtn_click_evt)

        self._addrange = ControlButton('Add interval', default=self.__rangelst_add_evt, visible=False)

        self._graph = GraphAreaWin(parent_win=self)

        super().__init__(*args, **kwargs)

        self.set_margin(10)

        self.formset = [
            ('_video', '_session', '_savebtn'),
            '_player',
            '=',
            ('_nblobs', '_togglegraph', '_chcksegm', ' '),
            ('_intensity', '_bgsub'),
            ('_area', '_resreduct'),
            ('_range', '_rangelst', '_addrange', '_multiple_range'),
            ('_applyroi', ' '),
            ('_rectbtn', '_polybtn', '_circlebtn', ' '),
            '_roi',
            ('_no_ids', '_pre_processing', '_progress', '_editpaths')
        ]

        self._graph.on_draw = self.__graph_on_draw_evt
        self._applyroi.changed_event = self.__apply_roi_changed_evt
        self._session.changed_event = self.__session_changed_evt
        self._player.drag_event = self.on_player_drag_in_video_window
        self._player.end_drag_event = self.on_player_end_drag_in_video_window
        self._player.click_event = self.on_player_click_in_video_window
        self._video.changed_event = self.__video_changed_evt
        self._video_path.changed_event = self.__video_changed_evt
        self._intensity.changed_event = self._player.refresh
        self._player.double_click_event = self.on_player_double_click_in_video_window
        self._player.process_frame_event = self.process_frame_evt
        self._multiple_range.changed_event = self.__multiple_range_changed_evt

        self.setMinimumHeight(900)
        self._player.setMinimumHeight(300)
        self._togglegraph.form.setMaximumWidth(180)
        self._roi.setMaximumHeight(100)
        self._session.form.setMaximumWidth(250)
        self._multiple_range.form.setMaximumWidth(130)

        self.__apply_roi_changed_evt()
        self.__session_changed_evt()

        self._multiple_files = False # flag to open multiple video files with similar names

    def save_form(self, data={}, path=None):
        data['open-multiple-files'] = self._multiple_files
        return super().save_form(data, path)



    def set_controls_enabled(self, status):
        super().set_controls_enabled(status)
        self._editpaths.enabled = status
        self._pre_processing.enabled = status
        self._savebtn.enabled = status
        self._player.enabled = status
        self._polybtn.enabled = status
        self._rectbtn.enabled = status
        self._circlebtn.enabled = status
        self._togglegraph.enabled = status
        self._addrange.enabled = status


    def process_frame_evt(self, frame):
        """
        Function called before an image is shown in the player.
        It does the pre-visualization segmentation and ROIs selection.
        """
        min_thresh, max_thresh = self._intensity.value
        min_area,   max_area   = self._area.value

        original_size = frame.shape[1], frame.shape[0]
        reduction = self._resreduct.value
        frame = cv2.resize(frame, None, fx=reduction, fy=reduction, interpolation=cv2.INTER_AREA)


        # Convert the frame to black & white
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape)>2 else frame
        mask = self.create_mask(*gray.shape)
        av_intensity = np.float32(np.mean(np.ma.array(gray, mask=mask==0)))
        av_frame     = gray / av_intensity


        bin_frame    = segment_frame( av_frame, min_thresh, max_thresh, self._background_img, mask, self._bgsub.value)
        boxes, mini_frames, _, areas, _, good_cnt, _ = blob_extractor(bin_frame.copy(), frame, int(min_area), int(max_area))
        self._detected_areas = areas
        #if self._nblobs.value<len(areas):
        #    self._nblobs.value = len(areas)

        cv2.drawContours(frame, good_cnt, -1, color=(0,0,255), thickness=-1)

        if conf.PYFORMS_MODE == 'GUI' and self._togglegraph.value:
            self._graph.draw()

        # The resize to the original size is required because of the draw of the ROI.
        frame = cv2.resize(frame, original_size, interpolation=cv2.INTER_AREA)

        self.draw_rois(frame)
        return frame

    def __rangelst_add_evt(self):
        win = RangeWin(
            parent_win=self,
            maximum=self._player.max,
            begin=self._player.video_index,
            control_list=self._rangelst
        )
        win.show()

    def __multiple_range_changed_evt(self):
        if self._multiple_range.value:
            self._range.hide()
            self._addrange.show()
            self._rangelst.show()
        else:
            self._range.show()
            self._addrange.hide()
            self._rangelst.hide()

    def __video_changed_evt(self):
        """
        Function called when the video file is selected.
        Ask to the player to load the video file.
        """
        if self.video_path:
            self._player.value = self.video_path
            if self._player.value:
                self._range.max    = self._player.max
                self._range.value  = [0, self._player.max]
                self.set_controls_enabled(True)
                self._multiple_files = self._player.multiple_files
            else:
                self.set_controls_enabled(False)

        else:
            self.set_controls_enabled(False)

    def __toggle_graph_evt(self):
        if self._togglegraph.value:
            self._graph.show()
        else:
            self._graph.hide()

    def __graph_on_draw_evt(self, figure):
        areas = self._detected_areas

        if not areas: return

        axes = figure.add_subplot(111)
        axes.clear()
        axes.bar(range(1,len(areas)+1), areas, 0.5)
        if len(areas) > 0:
            min_area = np.min(areas)
            axes.axhline(min_area, color = 'k', linewidth = .3)
            axes.set_title( str(len(areas)) + " blobs detected. Minimum area: " + str(min_area) )
            axes.set_xlabel('Blob index')
            axes.set_ylabel('Area in pixels')

    def __apply_roi_changed_evt(self):
        """
        Hide and show the controls to setup the background
        """
        if self._applyroi.value:
            self._roi.show()
            self._polybtn.show()
            self._rectbtn.show()
            self._circlebtn.show()
        else:
            self._roi.hide()
            self._polybtn.hide()
            self._rectbtn.hide()
            self._circlebtn.hide()

    def __session_changed_evt(self):
        video_folder   = os.path.dirname(self.video_path)
        session_folder = "session_{0}".format(self._session.value)
        session_path   = os.path.join(video_folder, session_folder)
        blobs_path_1   = os.path.join(session_path, 'preprocessing', 'blobs_collection_no_gaps.npy')
        blobs_path_2   = os.path.join(session_path, 'preprocessing', 'blobs_collection.npy')
        vidobj_path    = os.path.join(session_path, 'video_object.npy')

        if os.path.exists(session_path) and \
                (os.path.exists(blobs_path_1) or os.path.exists(blobs_path_2)) and \
                os.path.exists(vidobj_path):
            self._editpaths.enabled = True

        else:
            self._editpaths.enabled = False

    def __update_progress_evt(self, progress_count, max_count=None):
        if max_count is not None:
            self._progress.max = max_count
            self._progress.value = 0
            self._progress.show()
        elif self._progress.max == progress_count:
            #self._progress.hide()
            self._progress.value = 0
        else:
            self._progress.value = progress_count

    def __open_videoannotator_evt(self):

        video_folder   = os.path.dirname(self.video_path)
        session_folder = "session_{0}".format(self._session.value)
        session_path   = os.path.join(video_folder, session_folder)

        subprocess.Popen([sys.executable,'-m', 'pythonvideoannotator', session_path])
