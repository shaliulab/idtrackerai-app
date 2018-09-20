import numpy as np, cv2, math, os, pickle, logging

from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlCheckBox
from pyforms.controls import ControlCombo
from pyforms.controls import ControlFile
from pyforms.controls import ControlDir
from pyforms.controls import ControlPlayer
from pyforms.controls import ControlBoundingSlider
from pyforms.controls import ControlButton
from pyforms.controls import ControlMatplotlib
from pyforms.controls import ControlNumber
from pyforms.controls import ControlProgress

from idtrackerai.utils.video_utils import segment_frame, blob_extractor, cumpute_background

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





class IdTrackerAiGUI(BaseWidget):


    def __init__(self, *args, **kwargs):
        super().__init__(title='idtracker.ai')

        self.set_margin(10)
        #self.setMinimumHeight(800)

        self._session   = ControlText('Session', default='session0')
        self._video     = ControlFile('File')

        self._save_summaries = ControlCheckBox('Save tensorboard summaries')
        self._learning_rate  = ControlNumber('Learning rate', default=0.005, decimals=3)
        self._dropout_ratio  = ControlNumber(
            'Dropout ratio. If 1.0, no dropout is performed',
            default=1.0,
            helptext='For fully connected layers excluding softmax',
            decimals=1
        ) 
        self._optimiser       = ControlCheckBox('Optimiser. Acceptable optimisers: SGD and Adam', default='SGD')
        self._layers_to_train = ControlCombo('Layers to train')
        self._transfer_folder = ControlDir(
            'Knowlegde transfer folder', 
            helptext='Path to load convolutional weights from a pre-trained model'
        )
        self._tracking = ControlButton('Start protocol cascade', default=self.step2_tracking)
        

        self.formset = [
            '_session',
            '_video',
            '_save_summaries',
            '_learning_rate',
            '_dropout_ratio',
            '_optimiser',
            '_layers_to_train',
            '_transfer_folder',
            '_tracking',
            ' '
        ]

        self._layers_to_train.add_item('all')
        self._layers_to_train.add_item('fully')

        self._video.value = '/home/ricardo/bitbucket/idtracker-project/idtrackerai_video_example.avi'

        self._learning_rate.decimals = 3
        self._dropout_ratio.decimals = 1
        

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


#Execute the application
if __name__ == "__main__":
    import logging, locale
    
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s \t %(name)-50s %(message)s")
    
    from pyforms_gui.appmanager import start_app
    
    start_app( IdTrackerAiGUI, geometry=(2800,100,800, 600) )