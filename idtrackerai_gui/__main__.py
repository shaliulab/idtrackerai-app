

#Execute the application
if __name__ == "__main__":
    import logging, locale, coloredlogs

    coloredlogs.install(level='DEBUG', fmt='[%(levelname)-8s]   %(name)-50s %(message)s')
    
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)-8s]   %(name)-50s %(message)s")
    
    from .window_preprocessing  import IdTrackerAiGUI
    from pyforms_gui.appmanager import start_app
    
    start_app( IdTrackerAiGUI, geometry=(2800,100,800, 600) )
