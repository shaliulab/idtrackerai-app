import numpy as np, cv2, math, os, pickle, logging

from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlCheckBox
from pyforms.controls import ControlFile
from pyforms.controls import ControlPlayer
from pyforms.controls import ControlNumber
from pyforms.controls import ControlBoundingSlider
from pyforms.controls import ControlButton
from pyforms.controls import ControlMatplotlib
from pyforms.controls import ControlNumber
from pyforms.controls import ControlProgress

from idtrackerai.utils.video_utils import segment_frame, blob_extractor, cumpute_background
from idtrackerai.utils.py_utils import  getExistentFiles
from idtrackerai.constants import PROCESSES

#from idtrackerai.preprocessing.pre_processing import step1_pre_processing
#from idtrackerai.preprocessing.pre_processing import step2_tracking
#from idtrackerai.preprocessing.pre_processing import protocol1
from idtrackerai.list_of_fragments            import ListOfFragments
from idtrackerai.list_of_global_fragments     import ListOfGlobalFragments

from idtrackerai.video import Video

from idtrackerai.gui.tracker_api import TrackerAPI
from idtrackerai.gui.preprocessing_preview_api import PreprocessingPreviewAPI

from .helpers import Chosen_Video

logger = logging.getLogger(__name__)

import tensorflow as tf
with tf.Session() as sess:
    logger.info( "TENSORFLOW DEVICES: "+str(sess.list_devices()) )

def points_distance(p1, p2):
    return  math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def create_rectangle_points(start, end):
    return [ start, (end[0],start[1]), end, (start[0],end[1]) ]

def create_ellipse_points( start, end ):
    width = end[0]-start[0]
    height = end[1]-start[1]
    center = ( start[0] + width/2, start[1] + height/2 )

    distance = points_distance(start, end )
    nPoints = distance / 30
    if nPoints<8:nPoints = 8.0

    points = []
    for angleR in np.arange(0, math.pi*2, math.pi/nPoints):
        x = int(round(center[0] + width/2 * np.cos(angleR) ))
        y = int(round(center[1] + height/2 * np.sin(angleR)))
        points.append( ( x,y) )
    return points



class IdTrackerAiGUI(BaseWidget):


    def __init__(self, *args, **kwargs):
        super().__init__(title='idtracker.ai')

        self.set_margin(10)
        self.setMinimumHeight(800)

        self._session   = ControlText('Session', default='session0')
        self._video     = ControlFile('File', changed_event=self.__video_changed_evt)
        self._applyroi  = ControlCheckBox('Apply ROI?')
        self._roi       = ControlText('ROI')
        self._player    = ControlPlayer('Player')

        self._bgsub     = ControlCheckBox('Subtract background', changed_event=self.__bgsub_changed_evt)
        self._chcksegm  = ControlCheckBox('Check segmentation')
        self._resreduct = ControlNumber('Resolution reduction', default=1., minimum=0, maximum=1, decimals=2, step=0.1)
        # self._resreduct = ControlText('Resolution reduction', detault='1.')

        self._intensity = ControlBoundingSlider('Intensity', default=[0,135], min=0, max=255, changed_event=self._player.refresh)
        self._area      = ControlBoundingSlider('Area',      default=[150,60000], min=0, max=60000)
        self._range     = ControlBoundingSlider(None,        default=[0,10], min=0, max=255)
        self._circlebtn = ControlButton('Circle',    checkable=True)
        self._rectbtn   = ControlButton('Rectangle', checkable=True)
        self._nblobs    = ControlNumber('Detected blobs', default=8)
        self._graph     = ControlMatplotlib('Blobs area', toolbar=False, on_draw=self.__graph_on_draw_evt)
        self._progress  = ControlProgress('Progress')

        self._pre_processing = ControlButton('Pre processing', default=self.step1_pre_processing)
        self._tracking       = ControlButton('Start protocol cascade', default=self.step2_tracking)


        self.formset = [
            '_session',
            '_video',
            '_player',
            '_range',
            ('_circlebtn','_rectbtn',' ', ' '),
            ('_applyroi','_roi'),
            ('_bgsub','_chcksegm','_resreduct', ' '),
            '_intensity',
            '_area',
            '_nblobs',
            '_graph',
            ('_pre_processing', '_tracking'),
            '_progress'
        ]

        self._player.drag_event          = self.on_player_drag_in_video_window
        self._player.end_drag_event      = self.on_player_end_drag_in_video_window
        self._player.click_event         = self.on_player_click_in_video_window
        self._player.double_click_event  = self.on_player_double_click_in_video_window
        self._player.process_frame_event = self.process_frame_evt

        self._selected_point = None
        self._start_point    = None
        self._end_point      = None

        self._video.value = '/home/ricardo/bitbucket/idtracker-project/idtrackerai_video_example.avi'

    def __bgsub_changed_evt(self):
        if self._bgsub.value:
            # video      = cv2.VideoCapture(self._video.value)
            # background = cumpute_background(video)
            pass
        else:
            pass



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
        self._player.value = self._video.value
        self._range.max    = self._player.max
        self._range.value  = [0, self._player.max]


    def __draw_rois(self, frame):
        """
        Draw the ROIs lines in the frame.
        """
        points = self._roi.value
        try:
            points = eval( self._roi.value )
            cv2.polylines(frame, [np.array(points,np.int32)], True, (0,255,0), 2, lineType=cv2.LINE_AA)
            for i, point in enumerate( points ):
                if self._selected_point == i:
                    cv2.circle(frame, point, 4, (0,0,255), 2)
                else:
                    cv2.circle(frame, point, 4, (0,255,0), 2)
        except:
            pass

        if self._start_point and self._end_point:
            if self._rectbtn.checked:
                cv2.rectangle(frame, self._start_point, self._end_point, (233,44,44), 1 )
            elif self._circlebtn.checked and self._end_point[0]>self._start_point[0] and self._end_point[1]>self._start_point[1]:
                width = self._end_point[0]-self._start_point[0]
                height = self._end_point[1]-self._start_point[1]
                center = ( self._start_point[0] + width/2, self._start_point[1] + height/2 )
                cv2.ellipse( frame, (center, (width,height), 0), (233,44,44), 1 )

        return frame

    def __create_mask(self, height, width):
        """
        Create a mask based on the selected ROIs
        """
        points = self._roi.value
        mask   = np.zeros( (height, width) , dtype=np.uint8)
        if points:
            points = eval( self._roi.value )
            mask   = cv2.fillPoly(mask, [np.array(points,np.int32)], (255,255,255))
        else:
            mask = mask + 255

        return mask


    def process_frame_evt(self, frame):
        """
        Function called before an image is shown in the player.
        It does the pre-visualization segmentation and ROIs selection.
        """
        min_thresh, max_thresh = self._intensity.value
        min_area,   max_area   = self._area.value

        reduction = self._resreduct.value
        frame = cv2.resize(frame, None, fx=reduction, fy=reduction, interpolation=cv2.INTER_AREA)


        # Convert the frame to black & white
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape)>2 else frame
        mask = self.__create_mask(*gray.shape)

        av_intensity = np.float32(np.mean(gray))
        av_frame     = gray / av_intensity

        bin_frame    = segment_frame( av_frame, min_thresh, max_thresh, None, mask, False)
        boxes, mini_frames, _, areas, _, good_cnt, _ = blob_extractor(bin_frame.copy(), frame, int(min_area), int(max_area))
        self._detected_areas = areas
        if self._nblobs.value<len(areas):
            self._nblobs.value = len(areas)

        cv2.drawContours(frame, good_cnt, -1, color=(0,0,255), thickness=-1)

        self.__draw_rois(frame)

        self._graph.draw()
        return frame

    def select_point(self,x, y):
        try:
            coord  = ( x, y )
            points = eval(self._roi.value)
            for i, point in enumerate( points):
                if points_distance( coord, point ) <= 5:
                    self._selected_point = i
                    return
            self._selected_point = None
        except:
            pass


    def get_intersection_point_distance(self, test_point, point1, point2):
        p1 = np.float32(point1)
        p2 = np.float32(point2)
        p3 = np.float32(test_point)
        dist = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        return dist


    def on_player_double_click_in_video_window(self, event, x, y):
        mouse = ( int(x), int(y) )
        distances = []

        try:
            points   = list(eval(self._roi.value))
            n_points = len(points)
            for point_index, point in enumerate( points ):
                next_point = points[ (point_index+1) % n_points ]
                distance = self.get_intersection_point_distance(mouse, point, next_point )

                if distance<=5:
                    vector = next_point[0]-point[0], next_point[1]-point[1]
                    center = point[0]+vector[0]/2,point[1]+vector[1]/2
                    radius = points_distance(center, point)
                    mouse_distance = points_distance(center, mouse)
                    if mouse_distance<radius:
                        distances.append( (distance, point_index) )
        except:
            pass

        if len(distances)>0:
            distances = sorted(distances, key=lambda x: x[0])
            point_index = distances[0][1]
            points.insert( point_index + 1, mouse )

            self._roi.value = str(points)[1:-1]

            self._selected_point = point_index + 1

            if not self._player.is_playing: self._player.refresh()


    def on_player_click_in_video_window(self, event, x, y):
        self._selected_point = None

        if not self._rectbtn.checked and not self._circlebtn.checked:
            self.select_point( int(x), int(y) )


    def on_player_drag_in_video_window(self, start_point, end_point):
        self._start_point = ( int(start_point[0]), int(start_point[1]) )
        self._end_point   = ( int(end_point[0]), int(end_point[1]) )

        if self._selected_point!=None:
            try:
                points = list(eval(self._roi.value))
                points[self._selected_point] = self._end_point
                self._roi.value = str(points)[1:-1]
            except Exception as e:
                print(e)

        if not self._player.is_playing: self._player.refresh()

    def on_player_end_drag_in_video_window(self, start_point, end_point):
        self._start_point = int(start_point[0]), int(start_point[1])
        self._end_point   = int(end_point[0]),   int(end_point[1])

        points = None
        if self._rectbtn.checked:
            points = create_rectangle_points(self._start_point, self._end_point)
        elif self._circlebtn.checked and self._end_point[0]>self._start_point[0] and self._end_point[1]>self._start_point[1]:
            points = create_ellipse_points(self._start_point, self._end_point)

        if points: self._roi.value = str(points)[1:-1]

        self._start_point    = None
        self._end_point      = None
        self._rectbtn.checked   = False
        self._circlebtn.checked = False

        if not self._player.is_playing: self._player.refresh()


    def step1_pre_processing(self):

        video_object = Video(
            video_path=self._video.value
        )
        video_object.get_info()

        video_object._min_threshold = self._intensity.value[0]
        video_object._max_threshold = self._intensity.value[1]

        video_object._min_area = self._area.value[0]
        video_object._max_area = self._area.value[1]

        video_object._video_folder = os.path.dirname(self._video.value)
        video_object._subtract_bkg = self._bgsub.value
        video_object._number_of_animals = int(self._nblobs.value)
        video_object._apply_ROI = self._applyroi.value

        video_object.resolution_reduction = self._resreduct.value
        video_object._number_of_channels  = 1
        video_object._ROI = self.__create_mask(video_object.height, video_object.width)

        video_object.create_session_folder(self._session.value)

        logger.debug("create Chosen_Video")
        chosen_video = Chosen_Video(video=video_object)
        logger.debug("before init PreprocessingPreviewAPI")
        pre = PreprocessingPreviewAPI( chosen_video )

        logger.debug("pre object: "+str(pre))

        logger.debug('call: init_preview')
        pre.init_preview()

        logger.debug('call: init_preproc_parameters')
        pre.init_preproc_parameters()

        logger.debug('call: segment')
        pre.segment(
            self._intensity.value[0],
            self._intensity.value[1],
            self._area.value[0],
            self._area.value[1]
        )

        logger.debug('call: compute_list_of_blobs')
        pre.compute_list_of_blobs()

        logger.debug('call: check_segmentation_consistency')
        pre.check_segmentation_consistency(self._chcksegm.value)

        logger.debug('call: save_list_of_blobs')
        pre.save_list_of_blobs()

        logger.debug('call: model_area_and_crossing_detector')
        pre.model_area_and_crossing_detector()

        logger.debug('call: train_and_apply_crossing_detector')
        pre.train_and_apply_crossing_detector()

        logger.debug('call: generate_list_of_fragments_and_global_fragments')
        pre.generate_list_of_fragments_and_global_fragments()

        trainner = pre.crossing_detector_trainer
        list_of_fragments = pre.list_of_fragments
        list_of_global_fragments = pre.list_of_global_fragments

        if trainner:

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
            fig.set_facecolor((.188, .188, .188))
            fig.subplots_adjust(left=0.1, bottom=0.15, right=.9, top=.95, wspace=None, hspace=1)
            fig.set_facecolor((.188, .188, .188))
            [(ax.set_facecolor((.188, .188, .188)), ax.tick_params(color='white', labelcolor='white'), ax.xaxis.label.set_color('white'), ax.yaxis.label.set_color('white')) for ax in ax_arr]
            [spine.set_edgecolor('white') for ax in ax_arr for spine in ax.spines.values()]
            trainner.store_training_accuracy_and_loss_data.plot(ax_arr, color = 'r', plot_now = False, legend_font_color = "white")
            trainner.store_validation_accuracy_and_loss_data.plot(ax_arr, color ='b', plot_now = False, legend_font_color = "white")
            plt.show()

            print('end')

        #protocol1(video, list_of_fragments, list_of_global_fragments)


    def step2_tracking(self):

        video_folder   = os.path.dirname(self._video.value)
        session_folder = "session_{0}".format(self._session.value)

        videoobj_filepath   = os.path.join(video_folder, session_folder, 'video_object.npy')
        fragments_filepath  = os.path.join(video_folder, session_folder, 'preprocessing', 'fragments.npy')
        gfragments_filepath = os.path.join(video_folder, session_folder, 'preprocessing', 'global_fragments.npy')

        video_object             = np.load(videoobj_filepath).item()
        list_of_fragments        = ListOfFragments.load(fragments_filepath)
        list_of_global_fragments = ListOfGlobalFragments.load(gfragments_filepath, list_of_fragments.fragments)

        chosen_video = Chosen_Video(
            video=video_object,
            list_of_fragments=list_of_fragments,
            list_of_global_fragments=list_of_global_fragments
        )

        chosen_video.existent_files, chosen_video.old_video = getExistentFiles(chosen_video.video, PROCESSES)

        tracker = TrackerAPI( chosen_video )

        tracker.init_tracking()
        tracker.protocol1()
        tracker.accumulate()
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
