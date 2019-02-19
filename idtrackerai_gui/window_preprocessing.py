import numpy as np, cv2, math, os, logging
from confapp import conf

logger = logging.getLogger(__name__)

from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlCheckBox
from pyforms.controls import ControlFile
from pyforms.controls import ControlList
from pyforms.controls import ControlPlayer
from pyforms.controls import ControlBoundingSlider
from pyforms.controls import ControlButton
from pyforms.controls import ControlNumber
from pyforms.controls import ControlProgress

from idtrackerai.utils.video_utils import segment_frame, blob_extractor, cumpute_background
from idtrackerai.utils.py_utils import  getExistentFiles
from idtrackerai.constants import PROCESSES

from idtrackerai.list_of_fragments            import ListOfFragments
from idtrackerai.list_of_global_fragments     import ListOfGlobalFragments

from idtrackerai.video import Video

from idtrackerai.gui.tracker_api import TrackerAPI
from idtrackerai.gui.preprocessing_preview_api import PreprocessingPreviewAPI

from .gui.roi_selection import ROISelectionWin
from .gui.grapharea_win import GraphAreaWin
from .gui.range_win import RangeWin

from .helpers import Chosen_Video

import tensorflow as tf
with tf.Session() as sess:
    logger.info( "TENSORFLOW DEVICES: "+str(sess.list_devices()) )





class IdTrackerAiGUI(BaseWidget, ROISelectionWin):


    def __init__(self, *args, **kwargs):
        BaseWidget.__init__(self, title='idtracker.ai')
        ROISelectionWin.__init__(self)

        self.set_margin(10)

        self._session   = ControlText('Session', default='session0')
        self._video     = ControlFile('File', changed_event=self.__video_changed_evt)
        self._applyroi  = ControlCheckBox('Apply ROI?', changed_event=self.__apply_roi_changed_evt, enabled=False)
        self._player    = ControlPlayer('Player', enabled=False, multiple_files=True)

        self._roi       = ControlList('ROI', enabled=False, readonly=True, select_entire_row=True,
                                      item_selection_changed_event=self.roi_selection_changed_evt,
                                      remove_function=self.remove_roi)
        self._polybtn   = ControlButton('Polygon', checkable=True, enabled=False, default=self.polybtn_click_evt)
        self._rectbtn   = ControlButton('Rectangle', checkable=True, enabled=False)
        self._circlebtn = ControlButton('Ellipse', checkable=True, enabled=False, default=self.circlebtn_click_evt)

        self._bgsub     = ControlCheckBox('Subtract background', changed_event=self.__bgsub_changed_evt, enabled=False)
        self._chcksegm  = ControlCheckBox('Check segmentation', enabled=False)
        self._resreduct = ControlNumber('Resolution reduction', default=1., minimum=0, maximum=1, decimals=2, step=0.1, enabled=False)

        self._intensity = ControlBoundingSlider('Threshold', default=[0,135], min=0, max=255, changed_event=self._player.refresh, enabled=False)
        self._area      = ControlBoundingSlider('Blobs area', default=[150,60000], min=0, max=60000, enabled=False)
        self._range     = ControlBoundingSlider('Frames range', default=[0,10], min=0, max=255, enabled=False)
        self._nblobs    = ControlNumber('N blobs', default=8, enabled=False)
        self._progress  = ControlProgress('Progress', enabled=False)

        self._addrange  = ControlButton('Add range', default=self.__rangelst_add_evt)
        self._rangelst  = ControlText( 'Frames ranges', visible=False )
        self._multiple_range = ControlCheckBox('Multiple ranges', enabled=False, changed_event=self.__multiple_range_changed_evt)

        self._togglegraph = ControlCheckBox('Graph', changed_event=self.__toggle_graph_evt)
        self._graph = GraphAreaWin(parent_win=self)

        if conf.PYFORMS_MODE=='GUI':
            self._pre_processing = ControlButton('Track video', default=self.track_video, enabled=False)
            self._savebtn        = ControlButton('Save parameters', default=self.save_window, enabled=False)


        self.formset = [
            ('_video','_session'),
            '_player',
            '=',
            ('_range','_rangelst','_addrange','_multiple_range'),
            '_intensity',
            ('_area', '_togglegraph'),
            ('_nblobs', '_resreduct', ' ', '_applyroi', '_chcksegm', '_bgsub'),
            ('_polybtn','_rectbtn', '_circlebtn', ' '),
            '_roi',
            ('_pre_processing', '_savebtn', '_progress')
        ]

        self.load_order = [
            '_session', '_video', '_range','_rangelst', '_multiple_range',
            '_intensity', '_area', '_nblobs',
            '_resreduct', '_chcksegm', '_roi',
            '_bgsub'
        ]

        self._graph.on_draw              = self.__graph_on_draw_evt
        self._player.drag_event          = self.on_player_drag_in_video_window
        self._player.end_drag_event      = self.on_player_end_drag_in_video_window
        self._player.click_event         = self.on_player_click_in_video_window
        self._player.double_click_event  = self.on_player_double_click_in_video_window
        self._player.process_frame_event = self.process_frame_evt


        if conf.PYFORMS_MODE=='GUI':
            self.setMinimumHeight(900)
            self._player.setMinimumHeight(300)
            self._togglegraph.form.setMaximumWidth(70)
            self._roi.setMaximumHeight(100)
            self._session.form.setMaximumWidth(250)
            self._multiple_range.form.setMaximumWidth(130)

        self._video.value = '/home/ricardo/bitbucket/idtracker-project/idtrackerai_video_example_01.avi'

        self.__apply_roi_changed_evt()
        self.__bgsub_changed_evt()

        #self._resreduct.value = 0.3
        #self._area.value = [5, 50]

        # store the computed background with the correct resolution reduction
        self._background_img = None
        # store the computed background with the original size
        self._original_bkg = None

    #########################################################
    ## GUI EVENTS ###########################################
    #########################################################

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

    def __toggle_graph_evt(self):
        if self._togglegraph.value:
            self._graph.show()
        else:
            self._graph.hide()

    def __bgsub_changed_evt(self):

        if self._bgsub.value:
            if self._player.value:
                video = Video( video_path=self._video.value )
                video.get_info()
                video._subtract_bkg = True
                video._original_bkg = cumpute_background(video)
                self._original_bkg = video.original_bkg
                video.resolution_reduction = self._resreduct.value
                self._background_img = video.bkg
            else:
                self._bgsub.value = False


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


    def __graph_on_draw_evt(self, figure):
        areas = self._detected_areas

        if not areas: return

        axes = figure.add_subplot(111)
        axes.clear()
        axes.bar(range(1,len(areas)+1), areas, 0.5)
        if len(areas) > 0:
            min_area = np.min(areas)
            axes.axhline(min_area, color = 'w', linewidth = .3)
            axes.set_title( str(len(areas)) + " blobs detected. Minimum area: " + str(min_area) )


    def __video_changed_evt(self):
        """
        Function called when the video file is selected.
        Ask to the player to load the video file.
        """
        if self._video.value:
            self._player.value = self._video.value
            if self._player.value:
                self._range.max    = self._player.max
                self._range.value  = [0, self._player.max]
                self.__set_enabled(True)

            else:
                self.__set_enabled(False)

        else:
            self.__set_enabled(False)


    def __set_enabled(self, status):
        self._applyroi.enabled = status
        self._roi.enabled = status
        self._player.enabled = status
        self._bgsub.enabled = status
        self._chcksegm.enabled = status
        self._resreduct.enabled = status
        self._intensity.enabled = status
        self._area.enabled = status
        self._range.enabled = status
        self._polybtn.enabled = status
        self._rectbtn.enabled = status
        self._circlebtn.enabled = status
        self._nblobs.enabled = status
        self._graph.enabled = status
        self._progress.enabled = status
        self._multiple_range.enabled = status
        self._rangelst.enabled = status
        self._addrange.enabled = status

        if conf.PYFORMS_MODE == 'GUI':
            self._pre_processing.enabled = status
            self._savebtn.enabled = status











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
        if self._nblobs.value<len(areas):
            self._nblobs.value = len(areas)

        cv2.drawContours(frame, good_cnt, -1, color=(0,0,255), thickness=-1)

        if conf.PYFORMS_MODE == 'GUI' and self._togglegraph.value:
            self._graph.draw()

        # The resize to the original size is required because of the draw of the ROI.
        frame = cv2.resize(frame, original_size, interpolation=cv2.INTER_AREA)

        self.draw_rois(frame)
        return frame

    def track_video(self):

        self._video.enabled=False
        self._session.enabled=False
        self.__set_enabled(False)

        self.step1_pre_processing()
        self.step2_tracking()

        self._video.enabled=True
        self._session.enabled=True
        self.__set_enabled(True)



    def step1_pre_processing(self):

        video_object = Video(
            video_path=self._video.value
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

        video_object._video_folder      = os.path.dirname(self._video.value)
        video_object._subtract_bkg      = self._bgsub.value
        video_object._original_bkg      = self._original_bkg
        video_object._number_of_animals = int(self._nblobs.value)
        video_object._apply_ROI         = self._applyroi.value
        video_object._original_ROI      = self.create_mask(video_object.original_height, video_object.original_width)

        video_object.resolution_reduction = self._resreduct.value

        video_object.create_session_folder(self._session.value)

        logger.debug("create Chosen_Video")
        chosen_video = Chosen_Video(video=video_object)
        logger.debug("before init PreprocessingPreviewAPI")
        pre = PreprocessingPreviewAPI( chosen_video )

        logger.debug("pre object: "+str(pre))

        self._progress.max = 9

        logger.debug('call: init_preview')
        pre.init_preview()
        self._progress.value = 1

        logger.debug('call: init_preproc_parameters')
        pre.init_preproc_parameters()
        self._progress.value = 2

        logger.debug('call: segment')
        pre.segment(
            self._intensity.value[0],
            self._intensity.value[1],
            self._area.value[0],
            self._area.value[1]
        )
        self._progress.value = 3

        video_object.create_images_folders() # for ram optimization
        logger.debug('call: compute_list_of_blobs')
        pre.compute_list_of_blobs()
        self._progress.value = 4

        logger.debug('call: check_segmentation_consistency')
        pre.check_segmentation_consistency(self._chcksegm.value)
        self._progress.value = 5

        logger.debug('call: save_list_of_blobs')
        pre.save_list_of_blobs()
        self._progress.value = 6

        logger.debug('call: model_area_and_crossing_detector')
        pre.model_area_and_crossing_detector()
        self._progress.value = 7

        logger.debug('call: train_and_apply_crossing_detector')
        pre.train_and_apply_crossing_detector()
        self._progress.value = 8

        logger.debug('call: generate_list_of_fragments_and_global_fragments')
        pre.generate_list_of_fragments_and_global_fragments()
        self._progress.value = 9

        trainner = pre.crossing_detector_trainer
        list_of_fragments = pre.list_of_fragments
        list_of_global_fragments = pre.list_of_global_fragments

        if chosen_video.video.there_are_crossings:

            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.rcParams.update({
                'font.size':        8,
                'axes.labelsize':   8,
                'xtick.labelsize' : 8,
                'ytick.labelsize' : 8,
                'legend.fontsize':  8
            })

            fig, ax_arr = plt.subplots(3)
            # fig.set_facecolor((.188, .188, .188))
            fig.subplots_adjust(left=0.1, bottom=0.15, right=.9, top=.95, wspace=None, hspace=1)
            # fig.set_facecolor((.188, .188, .188))
            # [(ax.set_facecolor((.188, .188, .188)), ax.tick_params(color='white', labelcolor='white'), ax.xaxis.label.set_color('white'), ax.yaxis.label.set_color('white')) for ax in ax_arr]
            # [spine.set_edgecolor('white') for ax in ax_arr for spine in ax.spines.values()]
            trainner.store_training_accuracy_and_loss_data.plot(ax_arr, color = 'r', plot_now = False)
            trainner.store_validation_accuracy_and_loss_data.plot(ax_arr, color ='b', plot_now = False)
            # plt.show()
            fig.savefig(os.path.join(video_object.crossings_detector_folder, 'output_crossing_dectector.pdf'))

    def step2_tracking(self):

        video_folder   = os.path.dirname(self._video.value)
        session_folder = "session_{0}".format(self._session.value)

        videoobj_filepath   = os.path.join(video_folder, session_folder, 'video_object.npy')
        fragments_filepath  = os.path.join(video_folder, session_folder, 'preprocessing', 'fragments.npy')
        gfragments_filepath = os.path.join(video_folder, session_folder, 'preprocessing', 'global_fragments.npy')

        video_object = np.load(videoobj_filepath).item()
        video_object.create_session_folder(self._session.value)

        if video_object.number_of_animals != 1:
            list_of_fragments        = ListOfFragments.load(fragments_filepath)
            list_of_global_fragments = ListOfGlobalFragments.load(gfragments_filepath, list_of_fragments.fragments)
        else:
            list_of_fragments        = None
            list_of_global_fragments = None

        chosen_video = Chosen_Video(
            video=video_object,
            list_of_fragments=list_of_fragments,
            list_of_global_fragments=list_of_global_fragments
        )

        chosen_video.existent_files, chosen_video.old_video = getExistentFiles(chosen_video.video, PROCESSES)

        tracker = TrackerAPI( chosen_video )

        tracker.start_tracking()
        """
        print(video_object)
        print(list_of_fragments)
        print(list_of_global_fragments)

        step2_tracking(video_object)

        protocol1(video_object, list_of_fragments, list_of_global_fragments)
        """

    def __update_progress(self, value, label=None, total=None):

        if total is not None: self._progress.max   = total
        if label is not None: self._progress.label = label

        self._progress.value = value
