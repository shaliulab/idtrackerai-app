from idtrackerai.video import Video

class Chosen_Video(object):
    
    def __init__(self, processes_list = None, **kwargs):
        self.video = kwargs.get('video',  Video() )
        self.list_of_fragments = kwargs.get('list_of_fragments', None)
        self.list_of_global_fragments = kwargs.get('list_of_global_fragments', None)

        self.chosen = 'Default String'
        self.processes_list = processes_list
        self.processes_to_restore = {}
        self.old_video = None
        

    def set_chosen_item(self, chosen_string):
        self.chosen = chosen_string
