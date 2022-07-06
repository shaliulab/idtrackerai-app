import numpy as np, os, logging
import warnings
from confapp import conf, load_config

from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlCheckBox
from pyforms.controls import ControlFile
from pyforms.controls import ControlList
from pyforms.controls import ControlBoundingSlider

from pyforms.controls import ControlNumber
from pyforms.controls import ControlProgress

try:
    import idtrackerai
except ModuleNotFoundError as error:
    import sys
    print("Cannot load idtrackerai")
    print(f"PYTHON PATH: {sys.path}")
    raise error


from idtrackerai.animals_detection.segmentation_utils import (
    compute_background,
)

from idtrackerai.video import Video


from idtrackerai.animals_detection import AnimalsDetectionAPI
from idtrackerai.crossings_detection import CrossingsDetectionAPI
from idtrackerai.fragmentation import FragmentationAPI
from idtrackerai.tracker.tracker import TrackerAPI

from .gui.roi_selection import ROISelectionWin
from .gui.setup_points_selection import SetupInfoWin
from .gui.player_win_interactions import PlayerWinInteractions

from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.list_of_global_fragments import (
    ListOfGlobalFragments,
    create_list_of_global_fragments
)


logger = logging.getLogger(__name__)
try:
    import local_settings # type: ignore

    conf += local_settings
except ImportError:
    logger.info("Local settings file not available.")

try:
    import imgstore.constants
    CHUNK = load_config(imgstore.constants).CHUNK
    logger.info(f"Selected chunk={CHUNK}")

except ModuleNotFoundError:
    CHUNK=0


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

        # Main objects of idtracker.ai
        self.video_object = None
        self.list_of_blobs = None
        self.list_of_fragments = None
        self.list_of_global_fragments = None

        # App user interaction items
        # Session folder
        self._session = ControlText("Session", default="test")
        # Path to the video
        self._video = ControlFile("Video")
        self._video_path = ControlFile(
            "Video file. Note: overwrite the _video "
            "parameter defined in the json. "
            "Nice to have to execute the "
            "application in a cluster environment"
        )
        # Apply a ROI option
        self._applyroi = ControlCheckBox("Apply ROI", enabled=False)
        # List of ROIs that together will build the mask
        self._roi = ControlList(
            "ROI",
            enabled=False,
            readonly=True,
            select_entire_row=True,
            item_selection_changed_event=self.roi_selection_changed_evt,
            remove_function=self.remove_roi,
        )
        # Apply background subtraction option
        self._bgsub = ControlCheckBox(
            "Subtract background",
            changed_event=self.bgsub_changed_evt,
            enabled=False,
        )
        # Check that the number of segmented blobs in each frame is equal or
        # lower than the number of animals in the video
        self._chcksegm = ControlCheckBox("Check segmentation", enabled=False)
        # Resolution reduction factor
        self._resreduct = ControlNumber(
            "Resolution reduction",
            default=conf.RES_REDUCTION_DEFAULT,
            minimum=0.01,
            maximum=1,
            decimals=2,
            step=0.1,
            enabled=False,
        )
        # Intensity thresholds for animal detection (segmentation)
        self._intensity = ControlBoundingSlider(
            "Intensity thresholds",
            default=[conf.MIN_THRESHOLD_DEFAULT, conf.MAX_THRESHOLD_DEFAULT],
            min=conf.MIN_THRESHOLD,
            max=conf.MAX_THRESHOLD,
            enabled=False,
        )
        # Area thresholds for animal detection (segmentation)
        self._area = ControlBoundingSlider(
            "Area thresholds",
            default=[conf.MIN_AREA_DEFAULT, conf.MAX_AREA_DEFAULT],
            min=conf.AREA_LOWER,
            max=conf.AREA_UPPER,
            enabled=False,
        )
        # Ranges of frames to track
        self._range = ControlBoundingSlider(
            "Tracking interval",
            default=[0, 508],
            min=0,
            max=255,
            enabled=False,
        )  ###TODO: Upate max to the number of frames in the video
        self._rangelst = ControlText("Tracking intervals", visible=False)
        self._multiple_range = ControlCheckBox("Multiple", enabled=False)
        # Number of animals in the video
        self._number_of_animals = ControlNumber(
            "Number of animals",
            default=conf.NUMBER_OF_ANIMALS_DEFAULT,
            enabled=False,
        )
        # Tracking without identification
        self._no_ids = ControlCheckBox("Track without identities")
        # Setup points to store together with the trajectories
        self._add_setup_info = ControlCheckBox("Add setup info", enabled=False)
        self._points_list = ControlList(
            "Setup info",
            enabled=False,
            readonly=True,
            select_entire_row=True,
            item_selection_changed_event=self.add_setup_info_changed_evt,
            remove_function=self.remove_setup_info,
        )
        # Progress bar
        self._progress = ControlProgress("Progress", enabled=False)

        # App attributes
        self.formset = [
            ("_video", "_session"),
            ("_range", "_rangelst", "_multiple_range"),
            "_intensity",
            "_area",
            (
                "_number_of_animals",
                "_resreduct",
                " ",
                "_applyroi",
                "_chcksegm",
                "_bgsub",
            ),
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
            "_number_of_animals",
            "_resreduct",
            "_chcksegm",
            "_roi",
            "_no_ids",
            "_applyroi",
            "_bgsub",
            "_add_setup_info",
            "_points_list",
        ]

        # store the computed background with the correct resolution reduction
        self._background_img = None
        self._mask_img = None

        # flag to open multiple video files with similar names
        self._multiple_files = False

        self._frame_height = None
        self._frame_width = None

    #########################################################
    ## GUI EVENTS ###########################################
    #########################################################

    def load_form(self, data, path=None):
        self._multiple_files = data.get("open-multiple-files", False)
        super().load_form(data, path)

    def bgsub_changed_evt(self):
        if self.video_object is not None:
            self.__get_bkg_model()

    def set_controls_enabled(self, status):
        self._applyroi.enabled = status
        self._roi.enabled = status
        self._bgsub.enabled = status
        self._chcksegm.enabled = status
        self._resreduct.enabled = status
        self._intensity.enabled = status
        self._area.enabled = status
        self._range.enabled = status
        self._number_of_animals.enabled = status
        self._progress.enabled = status
        self._multiple_range.enabled = status
        self._rangelst.enabled = status
        self._no_ids.enabled = status
        self._add_setup_info.enabled = status
        self._points_list.enabled = status


    def track_video(self):
        logger.info("Calling track_video")
        self._video.enabled = False
        self._session.enabled = False
        self.set_controls_enabled(False)

        try:
            # Init tracking manager
            self._step0_init_video_object()
            self._step1_get_user_defined_parameters()
            # Preprocessing
            # success will be False if there are more blobs than animals and
            # the user asked to check the segmentation consistency
            success = self._step2_pre_processing()
            # Training and identification and post processing
            if success:
                success = self._step3_tracking()
            if success:
                # This flag is important to register the smoke tests that work
                logger.info("Success")

        except Exception as e:
            self.save()
            logger.error(e, exc_info=True)
            self.critical(str(e), "Error")

        self._video.enabled = True
        self._session.enabled = True
        self.set_controls_enabled(True)


    def preprocessing(self):

        try:
            # Init tracking manager
            self._step0_init_video_object()
            self._step1_get_user_defined_parameters()
            # Preprocessing
            # success will be False if there are more blobs than animals and
            # the user asked to check the segmentation consistency
            success = self._step2_pre_processing()
            return success

        except Exception as e:
            logger.error(e, exc_info=True)
            self.critical(str(e), "Error")

        finally:
            try:
                self.save()
                logger.info("DONE 100%: Preprocessing")
            except Exception as error:
                warnings.warn("Could not save data. All preprocessing is lost")
                warnings.warn(error, stacklevel=2)

    def tracking(self):
        # Training and identification and post processing
        self.load()
        self._step1_get_user_defined_parameters()
        try:
            success = self._step3_tracking()
            if success:
                # This flag is important to register the smoke tests that work
                logger.info("Success")

        except Exception as e:
            logger.error(e, exc_info=True)
            self.critical(str(e), "Error")

        finally:
            self.save()
            logger.info("DONE 100%: Tracking")

    def load(self):
        ########################################
        # TODO
        # Ensure state is preserved!
        ########################################
        #
        # This function is essential if I want to run preprocess and tracking
        # asynchronously (i.e. not necessarily right after within the same process)
        # This makes sense because preprocess is CPU bound and tracking GPU-bound
        # which means they do not use the same resources

        logger.info("Loading objects to base_idtrackerai")
        video_path = os.path.join(
            os.path.dirname(self.video_path), conf.ANALYSIS_FOLDER, f"session_{str(self._session.value).zfill(6)}",
            "video_object.npy"
        )

        if not os.path.exists(video_path):
            raise Exception(f"{video_path} does not exist. Are you sure you have run preprocessing for that session?")
            


        self.video_object = np.load(video_path, allow_pickle=True).item()
        self.list_of_blobs=ListOfBlobs.load(self.video_object.blobs_path)
        if not self.list_of_blobs.blobs_are_connected:
            if conf.RECONNECT_BLOBS_FROM_CACHE:
                self.list_of_blobs.reconect_from_cache()
            else:
                self.list_of_blobs.compute_overlapping_between_subsequent_frames()

        self.list_of_fragments=ListOfFragments.load(self.video_object.fragments_path)
        # tracker=TrackerAPI(self.video_object, self.list_of_blobs, self.list_of_fragments)
        # self.list_of_global_fragments=tracker._get_global_fragments()


    def save(self):
        logger.info("Saving objects from base_idtrackerai")
        self.video_object.save()
        if self.list_of_blobs is not None:
            self.list_of_blobs.save(self.video_object.blobs_path)
        if self.list_of_fragments is not None:
            self.list_of_fragments.save(self.video_object.fragments_path)
        if self.list_of_global_fragments is not None:
            self.list_of_global_fragments.save(
                self.video_object.global_fragments_path,
                self.list_of_fragments.fragments,
            )

    def _step0_init_video_object(self):

        self._progress.max = 4

        if not os.path.exists(self.video_path):
            raise Exception(
                "The video you are trying to track does not exist or the "
                f"path to the video is wrong: {self.video_path}"
            )

        # INIT AND POPULATE VIDEO OBJECT WITH PARAMETERS
        if self.video_object is None:
            logger.info("START: INIT VIDEO OBJECT")
            self.video_object = Video(
                video_path=self.video_path,
                open_multiple_files=self.open_multiple_files,
                chunk=CHUNK
            )
            logger.info("FINISH: INIT VIDEO OBJECT")
        self.video_object.create_session_folder(self._session.value)

    def __get_tracking_interval(self):
        if self._multiple_range.value and self._rangelst.value:
            try:
                self._tracking_interval = eval(self._rangelst.value)
            except Exception as e:
                logger.fatal(e, exc_info=True)
                self._tracking_interval = [self._range.value]
        else:
            self._tracking_interval = [self._range.value]

    def __get_bkg_model(self):
        if self._bgsub.value:
            if self._background_img is None:
                # Asked for background subtraction but it is not computed
                logger.info("Computing the background model")
                self._mask_img = self.create_mask(
                    self.video_object.original_height,
                    self.video_object.original_width,
                )
                self._background_img = compute_background(
                    self.video_object.video_paths,
                    self.video_object._chunk,
                    self.video_object.original_height,
                    self.video_object.original_width,
                    self.video_object.video_path,
                    self._mask_img,
                    self.video_object.episodes_start_end,
                )
            else:
                logger.info("Storing previously computed background model")
        else:
            # Did not ask for background subtraction
            logger.info("No background model computed")
            self._background_img = None

    def __get_mask(self):
        if self._applyroi:
            self._mask_img = self.create_mask(
                self.video_object.original_height,
                self.video_object.original_width,
            )
        else:
            self._mask_img = np.ones(
                (
                    self.video_object.original_height,
                    self.video_object.original_width,
                )
            )

    def _step1_get_user_defined_parameters(self):
        # TODO: Make background subtraction depend on the tracking interval
        # Collect tracking interval before computing the background
        self.__get_tracking_interval()
        self.__get_mask()
        # The computation of the background has a computation of the mask
        # TODO: Separate better mask and bkg
        self.__get_bkg_model()
        # TODO: Separate user defined parameters and advanced parameters
        # There are other parameters that come form the local_settings.py
        # It would be great to store them all in a singe json file so we
        # can check all the parameters used for tracking
        user_defined_parameters = {
            "number_of_animals": int(self._number_of_animals.value),
            "min_threshold": self._intensity.value[0],
            "max_threshold": self._intensity.value[1],
            "min_area": self._area.value[0],
            "max_area": self._area.value[1],
            "check_segmentation": self._chcksegm.value,
            "tracking_interval": self._tracking_interval,
            "apply_ROI": self._applyroi.value,
            "rois": self._roi.value,
            "mask": self._mask_img,
            "subtract_bkg": self._bgsub.value,
            "bkg_model": self._background_img,
            "resolution_reduction": self._resreduct.value,
            "track_wo_identification": self._no_ids.value,
            "setup_points": self.create_setup_poitns_dict(),
            "sigma_gaussian_blurring": conf.SIGMA_GAUSSIAN_BLURRING,
            "knowledge_transfer_folder": conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN,
            "identity_transfer": False,
            "identification_image_size": None,
        }

        if conf.IDENTITY_TRANSFER:
            # TODO: the identification_image_size is not really passed by
            # the used but inferred from the knowledge transfer folder
            (
                user_defined_parameters["identity_transfer"],
                user_defined_parameters["identification_image_size"],
            ) = TrackerAPI.check_if_identity_transfer_is_possible(
                user_defined_parameters["number_of_animals"],
                conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN,
            )

        self.video_object._user_defined_parameters = user_defined_parameters

    def __output_segmentation_consistency_warning(self, outfile_path):
        self.warning(
            "On some frames it was found more blobs than "
            "animals, "
            "you can find the index of these frames in the file:"
            f"<p>{outfile_path}</p>"
            "<p>Please readjust the segmentation parameters and "
            "press 'Track video' again.</p>",
            "Found more blobs than animals",
        )
        self._progress.value = 0
        self._final_message = self.SEGMENTATION_CHECK_FINAL_MESSAGE

    def _step2_pre_processing(self):

        logger.info("START: ANIMAL DETECTION")
        animals_detector = AnimalsDetectionAPI(self.video_object)
        self.list_of_blobs = animals_detector()
        # Check segmentation consistency
        segmentation_consistent = animals_detector.check_segmentation()
        if not segmentation_consistent and self._chcksegm.value:
            outfile_path = animals_detector.save_inconsistent_frames()
            self.save()  # saves video_object
            self.__output_segmentation_consistency_warning(outfile_path)
            return False  # This will make the tracking finish
        self._progress.value = 1
        logger.info("FINISH: ANIMAL DETECTION")

        logger.info("START: CROSSING DETECTION")
        crossings_detector = CrossingsDetectionAPI(
            self.video_object, self.list_of_blobs
        )
        crossings_detector()
        self._progress.value = 2
        logger.info("FINISH: CROSSING DETECTION")

        logger.info("START: FRAGMENTATION")
        fragmentator = FragmentationAPI(self.video_object, self.list_of_blobs)
        self.list_of_fragments = fragmentator()
        self._progress.value = 3
        logger.info("FINISH: FRAGMENTATION")
        return True  # This will make the tracking continue

    def _step3_tracking(self):

        tracker = TrackerAPI(
            self.video_object, self.list_of_blobs, self.list_of_fragments
        )

        if self.video_object.user_defined_parameters[
            "track_wo_identification"
        ]:
            # START: FRAGMENTATION
            logger.info("START: TRACKING WITHOUT IDENTITIES")
            tracker.track_wo_identification()
            logger.info("FINISH: TRACKING WITHOUT IDENTITIES")
            self._final_message = (
                "Tracking without identities finished. "
                "No estimated accuracy computed."
            )
        else:
            if (
                self.video_object.user_defined_parameters["number_of_animals"]
                == 1
            ):
                logger.info("START: TRACKING SINGLE ANIMAL")
                tracker.track_single_animal()
                logger.info("FINISH: TRACKING SINGLE ANIMAL")

            else:
                tracker.track_multiple_animals()
                self.list_of_fragments.update_identification_images_dataset()

            if self.video_object.estimated_accuracy is None:
                self.video_object.compute_estimated_accuracy()

            logger.info(
                "\nEstimated accuracy: {}".format(
                    self.video_object.estimated_accuracy
                )
            )

            self.video_object.delete_data()

            self._final_message = (
                "Tracking finished with {0:.2f} "
                "estimated accuracy.".format(
                    self.video_object.estimated_accuracy * 100
                )
            )
        self._progress.value = 4
        return True

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
