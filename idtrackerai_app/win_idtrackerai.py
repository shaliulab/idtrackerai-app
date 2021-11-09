import os, subprocess, cv2, numpy as np, sys
import glob
from confapp import conf

from PyQt5.QtWidgets import QApplication
from AnyQt.QtWidgets import QMessageBox
from scipy import ndimage  # TODO: used to fill binary holes see below

from idtrackerai.video import Video
from idtrackerai.animals_detection.segmentation import _process_frame
from idtrackerai.postprocessing.individual_videos import (
    generate_individual_videos,
)
from idtrackerai.postprocessing.trajectories_to_video import (
    generate_trajectories_video,
)

from pyforms.controls import ControlButton
from pyforms.controls import ControlPlayer
from pyforms.controls import ControlCheckBox

from .base_idtrackerai import BaseIdTrackerAi
from .gui.grapharea_win import GraphAreaWin
from .gui.range_win import RangeWin

NEW_GUI_FORMSET = (
    [
        ("_video", "_session", "_savebtn"),
        ("_number_of_animals", "_toggle_blobs_area_info", "_chcksegm", " "),
        ("_intensity", "_bgsub"),
        ("_area", "_resreduct"),
        ("_range", "_rangelst", "_addrange", "_multiple_range"),
        ("_applyroi", " "),
        ("_rectbtn", "_polybtn", "_circlebtn", " "),
        "_roi",
        ("_no_ids", "_pre_processing", "_progress", "_validation"),
        ("_indiv_videos", "_traj_video", " "),
    ],
    "_player",
)
OLD_GUI_FORMSET = [
    ("_video", "_session", "_savebtn"),
    "_player",
    "=",
    ("_number_of_animals", "_toggle_blobs_area_info", "_chcksegm", " "),
    ("_intensity", "_bgsub"),
    ("_area", "_resreduct"),
    ("_range", "_rangelst", "_addrange", "_multiple_range"),
    ("_applyroi", " "),
    ("_rectbtn", "_polybtn", "_circlebtn", " "),
    "_roi",
    ("_no_ids", "_pre_processing", "_progress", "_validation"),
    ("_indiv_videos", "_traj_video", " "),
]


class IdTrackerAiGUI(BaseIdTrackerAi):

    INDIV_VIDEO_BTN_LABEL = "Generate individual videos"
    GEN_INDIV_VIDEO_BTN_LABEL = "Generating individual videos..."
    TRAJ_VIDEO_BTN_LABEL = "Generate video with trajectories"
    GEN_TRAJ_VIDEO_BTN_LABEL = "Generate video with trajectories..."
    FINAL_MESSAGE_DEFAULT = "idtracker.ai terminated."

    def __init__(self, *args, **kwargs):

        self._player = ControlPlayer(
            "Player", enabled=False, multiple_files=True
        )
        self._toggle_blobs_area_info = ControlCheckBox(
            "Segmented blobs info",
            changed_event=self.__toggle_graph_evt,
            enabled=False,
        )
        self._pre_processing = ControlButton(
            "Track video", default=self.track_video, enabled=False
        )
        self._savebtn = ControlButton(
            "Save parameters", default=self.save_window, enabled=False
        )

        self._polybtn = ControlButton(
            "Polygon",
            checkable=True,
            enabled=False,
            default=self.polybtn_click_evt,
        )
        self._rectbtn = ControlButton(
            "Rectangle",
            checkable=True,
            enabled=False,
            default=self.rectbtn_click_evt,
        )
        self._circlebtn = ControlButton(
            "Ellipse",
            checkable=True,
            enabled=False,
            default=self.circlebtn_click_evt,
        )

        self._add_points_btn = ControlButton(
            "Add setup points",
            checkable=True,
            enabled=False,
            default=self.add_points_dict_click_evt,
        )

        self._addrange = ControlButton(
            "Add interval", default=self.__rangelst_add_evt, visible=False
        )

        self._indiv_videos = ControlButton(
            self.INDIV_VIDEO_BTN_LABEL,
            default=self.__generate_individual_videos_evt,
            enabled=False,
        )
        self._traj_video = ControlButton(
            self.TRAJ_VIDEO_BTN_LABEL,
            default=self.__generate_trajectories_video_evt,
            enabled=False,
        )
        self._validation = ControlButton(
            "Validate trajectories",
            default=self.__open_videoannotator_evt,
            enabled=False,
        )

        self._final_message = self.FINAL_MESSAGE_DEFAULT

        self._graph = GraphAreaWin(parent_win=self)

        super().__init__(*args, **kwargs)

        self.set_margin(10)

        self.formset = (
            [
                ("_video", "_resreduct", " "),
                (
                    "_number_of_animals",
                    "_toggle_blobs_area_info",
                    "_chcksegm",
                    " ",
                ),
                "_intensity",
                "_area",
                ("_applyroi", "_bgsub", " "),
                ("_rectbtn", "_polybtn", "_circlebtn", " "),
                "_roi",
                ("_range", "_rangelst", "_addrange", "_multiple_range"),
                ("_add_setup_info", " "),
                ("_add_points_btn", " "),
                "_points_list",
                ("_session", "_savebtn", " "),
                ("_no_ids", "_pre_processing", "_progress"),
                ("_validation", "_traj_video", "_indiv_videos", " "),
            ],
            "_player",
        )

        self._graph.on_draw = self.__graph_on_draw_evt
        self._applyroi.changed_event = self.__apply_roi_changed_evt
        self._add_setup_info.changed_event = self.__add_setup_info_changed_evt
        self._session.changed_event = self.__session_changed_evt
        self._player.drag_event = self.on_player_drag_in_video_window
        self._player.end_drag_event = self.on_player_end_drag_in_video_window
        self._player.click_event = self.on_player_click_in_video_window
        self._video.changed_event = self.__video_changed_evt
        self._video_path.changed_event = self.__video_changed_evt
        self._intensity.changed_event = self._player.refresh
        self._player.double_click_event = (
            self.on_player_double_click_in_video_window
        )
        self._player.process_frame_event = self.process_frame_evt
        self._multiple_range.changed_event = self.__multiple_range_changed_evt
        self._resreduct.changed_event = self.__resreduct_changed_evt

        self.setMinimumHeight(conf.GUI_MINIMUM_HEIGHT)
        self.setMinimumWidth(conf.GUI_MINIMUM_WIDTH)
        self._video.form.setMinimumWidth(300)
        self._player.setMinimumHeight(300)
        self._player.setMinimumWidth(300)
        self._toggle_blobs_area_info.form.setMaximumWidth(180)
        self._roi.setMaximumHeight(100)
        self._session.form.setMaximumWidth(250)
        self._multiple_range.form.setMaximumWidth(130)

        self.__apply_roi_changed_evt()
        self.__add_setup_info_changed_evt()
        self.__session_changed_evt()

    def save_form(self, data={}, path=None):
        data["open-multiple-files"] = self.open_multiple_files
        return super().save_form(data, path)

    def track_video(self):
        super().track_video()
        self.info(self._final_message, "Ended")

    def set_controls_enabled(self, status):
        super().set_controls_enabled(status)
        self._validation.enabled = status
        self._indiv_videos.enabled = status
        self._traj_video.enabled = status
        self._pre_processing.enabled = status
        self._savebtn.enabled = status
        self._player.enabled = status
        self._polybtn.enabled = status
        self._rectbtn.enabled = status
        self._circlebtn.enabled = status
        self._toggle_blobs_area_info.enabled = status
        self._addrange.enabled = status

        self._add_points_btn.enabled = status

    def process_frame_evt(self, frame):
        """
        Function called before an image is shown in the player.
        It does the pre-visualization segmentation and ROIs selection.
        """
        # Save original shape to rescale if resolution reduction is applied
        original_size = frame.shape[1], frame.shape[0]  # (width, height)
        self._frame_width = original_size[0]
        self._frame_height = original_size[1]
        # TODO: check if bkgmodel needs to be updated because of new ROI
        self._mask_img = self.create_mask(
            self._frame_height, self._frame_width
        )
        animal_detection_parameters = {
            "number_of_animals": int(self._number_of_animals.value),
            "min_threshold": self._intensity.value[0],
            "max_threshold": self._intensity.value[1],
            "min_area": self._area.value[0],
            "max_area": self._area.value[1],
            "tracking_interval": None,
            "apply_ROI": self._applyroi.value,
            "rois": self._roi.value,
            "mask": self._mask_img,
            "subtract_bkg": self._bgsub.value,
            "bkg_model": self._background_img,
            "resolution_reduction": self._resreduct.value,
            "sigma_gaussian_blurring": conf.SIGMA_GAUSSIAN_BLURRING,
        }

        (boxes, _, _, areas, _, contours, _,) = _process_frame(
            frame,
            animal_detection_parameters,
            -1,  # Get frame_number from the Widget
            save_pixels="NONE",
            save_segmentation_image="NONE",
        )

        # Save detected areas to plot the bar plot of the blobs size in pixels
        self._detected_areas = areas
        # Update graph with areas of seglemted blobs
        if conf.PYFORMS_MODE == "GUI" and self._toggle_blobs_area_info.value:
            self._graph.draw()
        # Draw detected blobs in frame
        if animal_detection_parameters["resolution_reduction"] != 1:
            frame = cv2.resize(
                frame,
                None,
                fx=animal_detection_parameters["resolution_reduction"],
                fy=animal_detection_parameters["resolution_reduction"],
                interpolation=cv2.INTER_AREA,
            )
        cv2.drawContours(frame, contours, -1, color=(0, 0, 255), thickness=-1)
        # Resize to original size (ROI and setup points are in original size)
        frame = cv2.resize(frame, original_size, interpolation=cv2.INTER_AREA)
        # Draw ROIs in frame
        self.draw_rois(frame)
        # Draw setup points in frame
        self.draw_points_list(frame)
        return frame

    def __resreduct_changed_evt(self):
        self._player.refresh()

    def __rangelst_add_evt(self):
        win = RangeWin(
            parent_win=self,
            maximum=self._player.max,
            begin=self._player.video_index,
            control_list=self._rangelst,
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
                self._range.max = self._player.max
                self._range.value = [0, self._player.max]
                self.set_controls_enabled(True)
                self._player.forward_one_frame()
            else:
                self.set_controls_enabled(False)
            video = Video(self.video_path, self.open_multiple_files)
            self.video_object = video
        else:
            self.set_controls_enabled(False)

    def bgsub_changed_evt(self):
        super().bgsub_changed_evt()
        self._player.refresh()

    def __toggle_graph_evt(self):
        if self._toggle_blobs_area_info.value:
            self._graph.show()
        else:
            self._graph.hide()

    def __graph_on_draw_evt(self, figure):
        areas = self._detected_areas

        if not areas:
            return
        figure.gca().remove()
        axes = figure.add_subplot(111)
        axes.clear()
        axes.bar(range(1, len(areas) + 1), areas, 0.5)
        if len(areas) > 0:
            min_area = np.min(areas)
            axes.axhline(min_area, color="k", linewidth=0.3)
            axes.set_title(
                str(len(areas))
                + " blobs detected. Minimum area: "
                + str(min_area)
            )
            axes.set_xlabel("Blob index")
            axes.set_ylabel("Area in pixels")

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

    def __add_setup_info_changed_evt(self):
        """
        Hide and show the controls to setup the background
        """
        if self._add_setup_info.value:
            self._points_list.show()
            self._add_points_btn.show()
        else:
            self._points_list.hide()
            self._add_points_btn.hide()

    def __session_changed_evt(self):
        video_folder = os.path.dirname(self.video_path)
        session_folder = "session_{0}".format(self._session.value)
        session_path = os.path.join(video_folder, session_folder)
        blobs_path_1 = os.path.join(
            session_path, "preprocessing", "blobs_collection_no_gaps.npy"
        )
        blobs_path_2 = os.path.join(
            session_path, "preprocessing", "blobs_collection.npy"
        )
        vidobj_path = os.path.join(session_path, "video_object.npy")
        trajs_wo_path = os.path.join(session_path, "trajectories_wo_gaps")
        trajs_path = os.path.join(session_path, "trajectories")

        if (
            os.path.exists(session_path)
            and (os.path.exists(blobs_path_1) or os.path.exists(blobs_path_2))
            and os.path.exists(vidobj_path)
        ):
            self._validation.enabled = True
        else:
            self._validation.enabled = False

        if (
            os.path.exists(session_path)
            and (os.path.exists(trajs_path) or os.path.exists(trajs_wo_path))
            and os.path.exists(vidobj_path)
        ):
            self._indiv_videos.enabled = True
            self._traj_video.enabled = True
        else:
            self._indiv_videos.enabled = False
            self._traj_video.enabled = False

    def __update_progress_evt(self, progress_count, max_count=None):
        if max_count is not None:
            self._progress.max = max_count
            self._progress.value = 0
            self._progress.show()
        elif self._progress.max == progress_count:
            self._progress.value = 0
        else:
            self._progress.value = progress_count

    def __open_videoannotator_evt(self):

        video_folder = os.path.dirname(self.video_path)
        session_folder = "session_{0}".format(self._session.value)
        session_path = os.path.join(video_folder, session_folder)

        subprocess.Popen(
            [sys.executable, "-m", "pythonvideoannotator", session_path]
        )

    def __generate_individual_videos_evt(self):
        self._indiv_videos.enabled = False
        self._indiv_videos.label = self.GEN_INDIV_VIDEO_BTN_LABEL
        QApplication.processEvents()
        video_object, trajectories = get_video_object_and_trajectories(
            self.video_path, self._session.value
        )
        generate_individual_videos(video_object, trajectories)
        self._indiv_videos.enabled = True
        self._indiv_videos.label = self.INDIV_VIDEO_BTN_LABEL
        QApplication.processEvents()

    def __generate_trajectories_video_evt(self):
        self._traj_video.enabled = False
        self._traj_video.label = self.GEN_TRAJ_VIDEO_BTN_LABEL
        QApplication.processEvents()
        video_object, trajectories = get_video_object_and_trajectories(
            self.video_path, self._session.value
        )
        generate_trajectories_video(video_object, trajectories)
        self._traj_video.enabled = True
        self._traj_video.label = self.TRAJ_VIDEO_BTN_LABEL
        QApplication.processEvents()

    @property
    def open_multiple_files(self):
        return self._player.multiple_files


def get_video_object_and_trajectories(video_path, session_name):
    video_folder = os.path.dirname(video_path)
    session_folder = "session_{0}".format(session_name)
    session_path = os.path.join(video_folder, session_folder)
    trajs_wo_path = os.path.join(session_path, "trajectories_wo_gaps")
    trajs_path = os.path.join(session_path, "trajectories")
    traj_wo_ids_path = os.path.join(
        session_path, "trajectories_wo_identification"
    )

    video_object = np.load(
        os.path.join(session_path, "video_object.npy"), allow_pickle=True
    ).item()

    if os.path.exists(trajs_wo_path):
        trajectories_file = glob.glob(os.path.join(trajs_wo_path, "*.npy"))[-1]
    elif os.path.exists(trajs_path):
        trajectories_file = glob.glob(os.path.join(trajs_path, "*.npy"))[-1]
    elif os.path.exists(traj_wo_ids_path):
        trajectories_file = glob.glob(os.path.join(traj_wo_ids_path, "*.npy"))[
            -1
        ]

    trajectories = np.load(trajectories_file, allow_pickle=True).item()[
        "trajectories"
    ]

    return video_object, trajectories
