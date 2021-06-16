import numpy as np, os, logging

logger = logging.getLogger(__name__)

from confapp import conf

try:

    import local_settings

    conf += local_settings
except:
    logger.info("Local settings file not available.")

from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlCheckBox
from pyforms.controls import ControlFile
from pyforms.controls import ControlList
from pyforms.controls import ControlBoundingSlider

from pyforms.controls import ControlNumber
from pyforms.controls import ControlProgress

from idtrackerai.utils.segmentation_utils import cumpute_background
from idtrackerai.utils.py_utils import getExistentFiles
from idtrackerai.constants import PROCESSES

from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments

from idtrackerai.video import Video

from idtrackerai.tracker_api import TrackerAPI
from idtrackerai.preprocessing_preview_api import PreprocessingPreviewAPI

from .gui.roi_selection import ROISelectionWin


from .helpers import Chosen_Video

import tensorflow as tf

with tf.Session() as sess:
    logger.info("TENSORFLOW DEVICES: " + str(sess.list_devices()))


class BaseIdTrackerAi(BaseWidget, ROISelectionWin):

    ERROR_MESSAGE_DEFAULT = (
        "\n \nIf this error persists please open an issue at "
        "https://gitlab.com/polavieja_lab/idtrackerai or "
        "send an email to idtrackerai@gmail.com. "
        "Check the log file idtrackerai-app.log in your "
        "working directory and attach it to the issue."
    )

    def __init__(self, *args, **kwargs):
        BaseWidget.__init__(self, title="idtracker.ai")
        ROISelectionWin.__init__(self)

        self._session = ControlText("Session", default="test")
        self._video = ControlFile("Video")
        self._video_path = ControlFile(
            "Video file. Note: overwrite the _video parameter defined in the json. Nice to have to execute the application in a cluster environment"
        )
        self._applyroi = ControlCheckBox("Apply ROI", enabled=False)

        self._roi = ControlList(
            "ROI",
            enabled=False,
            readonly=True,
            select_entire_row=True,
            item_selection_changed_event=self.roi_selection_changed_evt,
            remove_function=self.remove_roi,
        )

        self._bgsub = ControlCheckBox(
            "Subtract background",
            changed_event=self.bgsub_changed_evt,
            enabled=False,
        )
        self._chcksegm = ControlCheckBox("Check segmentation", enabled=False)
        self._resreduct = ControlNumber(
            "Resolution reduction",
            default=conf.RES_REDUCTION_DEFAULT,
            minimum=0.01,
            maximum=1,
            decimals=2,
            step=0.1,
            enabled=False,
        )

        self._intensity = ControlBoundingSlider(
            "Intensity thresholds",
            default=[conf.MIN_THRESHOLD_DEFAULT, conf.MAX_THRESHOLD_DEFAULT],
            min=conf.MIN_THRESHOLD,
            max=conf.MAX_THRESHOLD,
            enabled=False,
        )
        self._area = ControlBoundingSlider(
            "Area thresholds",
            default=[conf.MIN_AREA_DEFAULT, conf.MAX_AREA_DEFAULT],
            min=conf.AREA_LOWER,
            max=conf.AREA_UPPER,
            enabled=False,
        )
        self._range = ControlBoundingSlider(
            "Tracking interval",
            default=[0, 508],
            min=0,
            max=255,
            enabled=False,
        )  ###TODO: Change max of frames range to the number of frames of the video
        self._nblobs = ControlNumber(
            "Number of animals",
            default=conf.NUMBER_OF_ANIMALS_DEFAULT,
            enabled=False,
        )
        self._progress = ControlProgress("Progress", enabled=False)

        self._rangelst = ControlText("Tracking intervals", visible=False)
        self._multiple_range = ControlCheckBox("Multiple", enabled=False)

        self._no_ids = ControlCheckBox("Track without identities")

        self.formset = [
            ("_video", "_session"),
            ("_range", "_rangelst", "_multiple_range"),
            "_intensity",
            "_area",
            ("_nblobs", "_resreduct", " ", "_applyroi", "_chcksegm", "_bgsub"),
            "_roi",
            ("_no_ids", "_progress"),
        ]

        self.load_order = [
            "_session",
            "_video",
            "_range",
            "_rangelst",
            "_multiple_range",
            "_intensity",
            "_area",
            "_nblobs",
            "_resreduct",
            "_chcksegm",
            "_roi",
            "_no_ids",
            "_bgsub",
        ]

        # self._video.value = '/home/ricardo/bitbucket/idtracker-project/idtrackerai_video_example.avi'

        self.bgsub_changed_evt()

        # self._resreduct.value = 0.3
        # self._area.value = [5, 50]

        # store the computed background with the correct resolution reduction
        self._background_img = None
        # store the computed background with the original size
        self._original_bkg = None
        # flag to open multiple video files with similar names
        self._multiple_files = False

    #########################################################
    ## GUI EVENTS ###########################################
    #########################################################

    def load_form(self, data, path=None):
        self._multiple_files = data.get("open-multiple-files", False)
        super().load_form(data, path)

    def bgsub_changed_evt(self):

        if self._bgsub.value:
            if self.video_path:
                video = Video(
                    video_path=self.video_path,
                    open_multiple_files=self.open_multiple_files,
                )
                video.get_info()
                video._subtract_bkg = True
                video._original_bkg = cumpute_background(video)
                self._original_bkg = video.original_bkg
                video.resolution_reduction = self._resreduct.value
                self._background_img = video.bkg
            else:
                self._bgsub.value = False

    def set_controls_enabled(self, status):
        self._applyroi.enabled = status
        self._roi.enabled = status
        self._bgsub.enabled = status
        self._chcksegm.enabled = status
        self._resreduct.enabled = status
        self._intensity.enabled = status
        self._area.enabled = status
        self._range.enabled = status
        self._nblobs.enabled = status
        self._progress.enabled = status
        self._multiple_range.enabled = status
        self._rangelst.enabled = status
        self._no_ids.enabled = status

    def track_video(self):

        self._video.enabled = False
        self._session.enabled = False
        self.set_controls_enabled(False)

        try:
            if self.step1_pre_processing():
                if self._no_ids.value:
                    self.step2_wo_tracking()
                else:
                    self.step2_tracking()
        except Exception as e:
            logger.error(e, exc_info=True)
            self.critical(str(e), "Error")

        self._video.enabled = True
        self._session.enabled = True
        self.set_controls_enabled(True)

    def step1_pre_processing(self):

        if not os.path.exists(self.video_path):
            raise Exception(
                "The video you are trying to track does not exist or the path to the video is wrong."
            )

        video_object = Video(
            video_path=self.video_path,
            open_multiple_files=self.open_multiple_files,
        )
        video_object.get_info()

        # define the thresholds ranges
        video_object._min_threshold = self._intensity.value[0]
        video_object._max_threshold = self._intensity.value[1]

        # define the areas range
        video_object._min_area = self._area.value[0]
        video_object._max_area = self._area.value[1]

        # define the video range
        if self._multiple_range.value and self._rangelst.value:
            try:
                video_object._tracking_interval = eval(self._rangelst.value)
            except Exception as e:
                logger.fatal(e, exc_info=True)
                video_object._tracking_interval = [self._range.value]
        else:
            video_object._tracking_interval = [self._range.value]

        video_object._video_folder = os.path.dirname(self.video_path)
        video_object._subtract_bkg = self._bgsub.value
        video_object._original_bkg = self._original_bkg
        video_object._number_of_animals = int(self._nblobs.value)
        video_object._apply_ROI = self._applyroi.value
        video_object._original_ROI = self.create_mask(
            video_object.original_height, video_object.original_width
        )

        video_object.resolution_reduction = self._resreduct.value

        # Check if it is identity transfer or knowledge_transfer
        if conf.IDENTITY_TRANSFER:
            video_object.check_and_set_identity_transfer_if_possible()

        video_object.create_session_folder(self._session.value)

        logger.debug("create Chosen_Video")
        chosen_video = Chosen_Video(video=video_object)
        logger.debug("before init PreprocessingPreviewAPI")
        pre = PreprocessingPreviewAPI(chosen_video)

        logger.debug("pre object: " + str(pre))

        self._progress.max = 9

        logger.debug("call: init_preview")
        pre.init_preview()
        self._progress.value = 1

        logger.debug("call: init_preproc_parameters")
        pre.init_preproc_parameters()
        self._progress.value = 2

        logger.debug("call: segment")
        pre.segment(
            self._intensity.value[0],
            self._intensity.value[1],
            self._area.value[0],
            self._area.value[1],
        )
        self._progress.value = 3

        video_object.create_images_folders()  # for ram optimization
        logger.debug("call: compute_list_of_blobs")
        pre.compute_list_of_blobs()
        self._progress.value = 4

        logger.debug("call: check_segmentation_consistency")

        if self._chcksegm.value and not pre.check_segmentation_consistency():
            outfile_path = os.path.join(
                video_object.session_folder, "inconsistent_frames.csv"
            )

            self.warning(
                "On some frames it was found more blobs than animals, "
                "you can find the index of these frames in the file:"
                "<p>{0}</p>"
                "<p>Please readjust the segmentation parameters and press 'Track video' again.</p>".format(
                    outfile_path
                ),
                "Found more blobs than animals",
            )
            with open(outfile_path, "w") as outfile:
                outfile.write(
                    "\n".join(
                        map(
                            str,
                            pre.chosen_video.video.frames_with_more_blobs_than_animals,
                        )
                    )
                )

            self._progress.value = 0
            self._final_message = "Readjust the segmentation parameters and track the video again."
            return False

        self._progress.value = 5

        logger.debug("call: save_list_of_blobs_segmented")
        pre.save_list_of_blobs_segmented()
        self._progress.value = 6

        logger.debug("call: model_area_and_crossing_detector")
        pre.model_area_and_crossing_detector()
        self._progress.value = 7

        logger.debug("call: train_and_apply_crossing_detector")
        pre.train_and_apply_crossing_detector()
        self._progress.value = 8

        logger.debug("call: generate_list_of_fragments_and_global_fragments")
        pre.generate_list_of_fragments_and_global_fragments()
        self._progress.value = 9

        trainner = pre.crossing_detector_trainer
        list_of_fragments = pre.list_of_fragments
        list_of_global_fragments = pre.list_of_global_fragments

        return True

    def step2_tracking(self):

        video_folder = os.path.dirname(self.video_path)
        session_folder = "session_{0}".format(self._session.value)

        videoobj_filepath = os.path.join(
            video_folder, session_folder, "video_object.npy"
        )
        fragments_filepath = os.path.join(
            video_folder, session_folder, "preprocessing", "fragments.npy"
        )
        gfragments_filepath = os.path.join(
            video_folder,
            session_folder,
            "preprocessing",
            "global_fragments.npy",
        )

        video_object = np.load(videoobj_filepath, allow_pickle=True).item()
        video_object.create_session_folder(self._session.value)

        if video_object.number_of_animals != 1:
            list_of_fragments = ListOfFragments.load(fragments_filepath)
            list_of_global_fragments = ListOfGlobalFragments.load(
                gfragments_filepath, list_of_fragments.fragments
            )
        else:
            list_of_fragments = None
            list_of_global_fragments = None

        chosen_video = Chosen_Video(
            video=video_object,
            list_of_fragments=list_of_fragments,
            list_of_global_fragments=list_of_global_fragments,
        )

        chosen_video.existent_files, chosen_video.old_video = getExistentFiles(
            chosen_video.video, PROCESSES
        )

        tracker = TrackerAPI(chosen_video)

        tracker.start_tracking()

        if video_object.number_of_animals != 1:
            list_of_fragments.update_identification_images_dataset()

        logger.info("Estimated accuracy: {}".format(video_object.overall_P2))

        video_object.delete_data()

        self._final_message = (
            "Tracking finished with {0:.2f} estimated accuracy.".format(
                video_object.overall_P2 * 100
            )
        )

    def step2_wo_tracking(self):

        video_folder = os.path.dirname(self.video_path)
        session_folder = "session_{0}".format(self._session.value)

        videoobj_filepath = os.path.join(
            video_folder, session_folder, "video_object.npy"
        )
        fragments_filepath = os.path.join(
            video_folder, session_folder, "preprocessing", "fragments.npy"
        )
        gfragments_filepath = os.path.join(
            video_folder,
            session_folder,
            "preprocessing",
            "global_fragments.npy",
        )

        video_object = np.load(videoobj_filepath, allow_pickle=True).item()
        video_object.create_session_folder(self._session.value)

        if video_object.number_of_animals != 1:
            list_of_fragments = ListOfFragments.load(fragments_filepath)
            list_of_global_fragments = ListOfGlobalFragments.load(
                gfragments_filepath, list_of_fragments.fragments
            )
        else:
            list_of_fragments = None
            list_of_global_fragments = None

        chosen_video = Chosen_Video(
            video=video_object,
            list_of_fragments=list_of_fragments,
            list_of_global_fragments=list_of_global_fragments,
        )

        chosen_video.existent_files, chosen_video.old_video = getExistentFiles(
            chosen_video.video, PROCESSES
        )

        tracker = TrackerAPI(chosen_video)

        tracker.track_wo_identities()

        video_object.delete_data()

        self._final_message = "Tracking without identities finished. No estimated accuracy computed."

    def __update_progress(self, value, label=None, total=None):

        if total is not None:
            self._progress.max = total
        if label is not None:
            self._progress.label = label

        self._progress.value = value

    def alert(self, msg, title=None):
        msg = msg + self.ERROR_MESSAGE_DEFAULT
        self.message(msg, title, msg_type="error")

    def critical(self, msg, title=None):
        msg = msg + self.ERROR_MESSAGE_DEFAULT
        self.message(msg, title, msg_type="error")

    @property
    def open_multiple_files(self):
        return self._multiple_files

    @property
    def video_path(self):
        return (
            self._video_path.value
            if self._video_path.value
            else self._video.value
        )
