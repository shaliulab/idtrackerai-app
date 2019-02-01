

#Execute the application
if __name__ == "__main__":
    import logging, locale, coloredlogs

    coloredlogs.install(
        level='DEBUG', 
        fmt='[%(levelname)-8s] %(name)-40s %(message)s',
        #stream=open("idtrackerai-gui.log", 'w')
    )
    
    from .window_preprocessing  import IdTrackerAiGUI
    from pyforms import start_app
    
    start_app( IdTrackerAiGUI, geometry=(2800,100,800, 600) )
