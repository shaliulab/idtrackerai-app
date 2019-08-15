import sys, os
sys.path.append(os.getcwd())


def start():
    import logging, locale, coloredlogs
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='idtrackerai-app.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    coloredlogs.install(
        level='DEBUG',
        fmt='[%(levelname)-8s] %(name)-40s %(message)s',
        #stream=open("idtrackerai-app.log", 'w')
    )
    coloredlogs.install(
        level='DEBUG',
        fmt='[%(levelname)-8s] %(name)-40s %(message)s',
        stream=open("idtrackerai-app.log", 'w')
    )

    #logging.getLogger('').addHandler(open("idtrackerai-app.log", 'w'))

    from pyforms import start_app
    from confapp import conf

    logging.getLogger('PyQt5').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    if conf.PYFORMS_MODE=='GUI':
        from .win_idtrackerai  import IdTrackerAiGUI as App
    else:
        from .base_idtrackerai  import BaseIdTrackerAi as App


    try:
        start_app( App, geometry=(100,100,800, 600) )
    except SystemExit:
        pass
    except Exception as e:
        logger.info(e, exc_info=True)
        import sys
        import traceback
        ex_type, ex, tb = sys.exc_info()
        traceback.print_exception(ex_type, ex, tb)
        print("\n")
        print("idtracker.ai quit unexpectedly.")
        print("If this error persists please open an issue at")
        print("https://gitlab.com/polavieja_lab/idtrackerai")
        print("or send an email to idtrackerai@gmail.com")
        print("Check the log file idtrackerai-gui.log in")
        print("your working directory and attach it to the issue.")


#Execute the application
if __name__ == "__main__":
    start()
