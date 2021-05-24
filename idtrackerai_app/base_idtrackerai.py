import numpy as np, os, logging

from confapp import conf

from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlCheckBox
from pyforms.controls import ControlFile
from pyforms.controls import ControlList
from pyforms.controls import ControlBoundingSlider

from pyforms.controls import ControlNumber
from pyforms.controls import ControlProgress

from idtrackerai.utils.segmentation_utils import cumpute_background
from idtrackerai.utils.py_utils import get_computed_processes
from idtrackerai.constants import PROCESSES

from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments

from idtrackerai.video import Video

from idtrackerai.tracker_api import TrackerAPI
from idtrackerai.preprocessing_api import PreprocessingAPI

from .gui.roi_selection import ROISelectionWin
from .gui.setup_points_selection import SetupInfoWin
from .gui.player_win_interactions import PlayerWinInteractions

from .helpers import Chosen_Video


logger = logging.getLogger(__name__)
try:
    import local_settings

    conf += local_settings
except ImportError:
    logger.info("Local settings file not available.")


class BaseIdTrackerAi(
    BaseWidget, PlayerWinInteractions, ROISelectionWin, SetupInfoWin
):

    ERROR_MESSAGE_DEFAULT = (
        "\n \nIf this error persists please open an issue at "
        "https://gitlab.com/polavieja_lab/idtrackerai or "
        "send an email to idtrackerai@gmail.com. "
        "Check the log file idtrackerai-app.log in your "
        "working directory and attach it to the issue."
    )

    SEGMENTATION_CHECK_FINAL_MESSAGE = (
        "Readjust the segmentation parameters and track the video again."
    )

    def __init__(self, *args, **kwargs):
        BaseWidget.__init__(self, title="idtracker.ai")
        ROISelectionWin.__init__(self)
        SetupInfoWin.__init__(self)

        self._session = ControlText("Session", default="test")
        self._video = ControlFile("Video")
        self._video_path = ControlFile(
            "Video file. Note: overwrite the _video "
            "parameter defined in the json. "
            "Nice to have to execute the "
            "application in a cluster environment"
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

        self._add_setup_info = ControlCheckBox("Add setup info", enabled=False)
        self._points_list = ControlList(
            "Setup info",
            enabled=False,
            readonly=True,
            select_entire_row=True,
            item_selection_changed_event=self.add_setup_info_changed_evt,
            remove_function=self.remove_setup_info,
        )

        self.formset = [
            ("_video", "_session"),
            ("_range", "_rangelst", "_multiple_range"),
            "_intensity",
            "_area",
            ("_nblobs", "_resreduct", " ", "_applyroi", "_chcksegm", "_bgsub"),
            "_roi",
            ("_no_ids", "_progress"),
            ("_add_setup_info", " "),
            "_points_list",
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
            "_applyroi",
            "_bgsub",
            "_add_setup_info",
            "_points_list",
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
                video.get_info_from_video_file()
                video._subtract_bkg = True
                video._original_ROI = self.create_mask(
                    video._original_height, video._original_width
                )
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
        self._add_setup_info.enabled = status
        self._points_list.enabled = status

    def track_video(self):

        self._video.enabled = False
        self._session.enabled = False
        self.set_controls_enabled(False)

        try:
            # Init tracking manager (chosen_video)
            chosen_video = self.step0_init()

            # Preprocessing
            success_preprocessing = self.step1_pre_processing(chosen_video)
            if success_preprocessing:
                # Training and identification
                self.step2_tracking(chosen_video)
                # Post processing

        except Exception as e:
            chosen_video.save()
            logger.error(e, exc_info=True)
            self.critical(str(e), "Error")

        self._video.enabled = True
        self._session.enabled = True
        self.set_controls_enabled(True)

    def step0_init(self):

        if not os.path.exists(self.video_path):
            raise Exception(
                "The video you are trying to track does not exist or the path to the video is wrong."
            )

        # INIT AND POPULATE VIDEO OBJECT WITH PARAMETERS
        logger.info("START: INIT VIDEO OBJECT")
        video_object = Video(
            video_path=self.video_path,
            open_multiple_files=self.open_multiple_files,
        )
        video_object._video_folder = os.path.dirname(self.video_path)

        # Gets video with, height frames per second from the video file.
        video_object.get_info_from_video_file()

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
        video_object._subtract_bkg = (
            self._bgsub.value
        )  ## TODO: Check when the background is computed
        video_object._original_bkg = self._original_bkg
        video_object._number_of_animals = int(self._nblobs.value)
        video_object._apply_ROI = self._applyroi.value
        video_object._original_ROI = self.create_mask(
            video_object.original_height, video_object.original_width
        )
        # TODO: Improve this, it is a bit obscure
        # Needs to be done after _original_ROI is set since the resolution_reduction
        # setter changes the size of the _ROI attribute based on the _original_ROI
        video_object.resolution_reduction = self._resreduct.value
        video_object._track_wo_identities = self._no_ids.value
        ## TODO: This step is also done when computing the background model
        video_object._setup_points = self.create_setup_poitns_dict()

        # Finished reading preprocessing parameters
        video_object._has_preprocessing_parameters = True

        if conf.IDENTITY_TRANSFER:
            video_object.check_and_set_identity_transfer_if_possible()

        if conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN:
            video_object._tracking_with_knowledge_transfer = True
        else:
            video_object._tracking_with_knowledge_transfer = False

        video_object.create_session_folder(self._session.value)

        logger.debug("create Chosen_Video")
        chosen_video = Chosen_Video(video=video_object)
        logger.info("FINISH: INIT VIDEO OBJECT")

        # TODO: Uncomment to start developping restoring (tracking will fail)
        # (
        #     chosen_video.processes_to_restore,
        #     chosen_video.old_video,
        # ) = get_computed_processes(chosen_video.video, PROCESSES)
        # logger.info(
        #     f"Processes to restore: {chosen_video.processes_to_restore}"
        # )
        return chosen_video

    def step1_pre_processing(self, chosen_video):

        logger.debug("before init PreprocessingAPI")
        self._progress.max = 9

        # PREPROCESSING
        pre = PreprocessingAPI(chosen_video)
        self._progress.value = 1

        # START: ANIMAL DETECTION
        logger.info("START: ANIMAL DETECTION")
        pre.detect_animals()
        self._progress.value = 2
        # Check segmentation consistency
        segmentation_consistent = pre.check_segmentation_consistency()
        if self._chcksegm.value and not segmentation_consistent:
            outfile_path = pre.save_inconsistent_frames()
            self.warning(
                "On some frames it was found more blobs than "
                "animals, "
                "you can find the index of these frames in the file:"
                "<p>{0}</p>"
                "<p>Please readjust the segmentation parameters and press 'Track video' again.</p>".format(
                    outfile_path
                ),
                "Found more blobs than animals",
            )
            self._progress.value = 0
            self._final_message = self.SEGMENTATION_CHECK_FINAL_MESSAGE
            return False
        self._progress.value = 3
        # Save list of blobs
        pre.save_list_of_blobs_segmented()
        self._progress.value = 4
        # FINISH: ANIMAL DETECTION
        logger.info("FINISH: ANIMAL DETECTION")

        # START: CROSSING DETECTION
        logger.info("START: CROSSING DETECTION")
        pre.compute_model_area()
        self._progress.value = 5
        pre.set_identification_images()
        self._progress.value = 6
        pre.connect_list_of_blobs()
        self._progress.value = 7
        logger.debug("call: train_and_apply_crossing_detector")
        pre.train_and_apply_crossing_detector()
        self._progress.value = 8
        # FINISH: CROSSING DETECTION
        logger.info("FINISH: CROSSING DETECTION")

        # START: FRAGMENTATION
        logger.info("START: FRAGMENTATION")
        pre.generate_list_of_fragments_and_global_fragments()
        self._progress.value = 9
        # FINISH: FRAGMENTATION
        logger.info("FINISH: FRAGMENTATION")
        return True

    def reinit_chosen_video(self):
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

    def step2_tracking(self, chosen_video=None):

        tracker = TrackerAPI(chosen_video)

        if chosen_video.video.track_wo_identities:
            # START: FRAGMENTATION
            logger.info("START: TRACKING WITHOUT IDENTITIES")
            tracker.track_wo_identities()
            logger.info("FINISH: TRACKING WITHOUT IDENTITIES")
            self._final_message = (
                "Tracking without identities finished. "
                "No estimated accuracy computed."
            )
        else:
            if chosen_video.video.number_of_animals == 1:
                logger.info("START: TRACKING SINGLE ANIMAL")
                tracker.track_single_animal()
                logger.info("FINISH: TRACKING SINGLE ANIMAL")

            else:
                if (
                    chosen_video.list_of_global_fragments.number_of_global_fragments
                    == 1
                ):
                    logger.info("START: TRACKING SINGLE GLOBAL FRAGMENT")
                    tracker.track_single_global_fragment_video()
                    logger.info("FINISH: TRACKING SINGLE GLOBAL FRAGMENT")

                else:
                    logger.info("START: TRACKING")
                    tracker.track_w_identities()
                    logger.info("FINISH: TRACKING")

                chosen_video.list_of_fragments.update_identification_images_dataset()

            logger.info(
                "Estimated accuracy: {}".format(chosen_video.video.overall_P2)
            )

            chosen_video.video.delete_data()

            self._final_message = (
                "Tracking finished with {0:.2f} "
                "estimated accuracy.".format(
                    chosen_video.video.overall_P2 * 100
                )
            )

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
