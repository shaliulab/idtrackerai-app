

#Execute the application
if __name__ == "__main__":
    import logging, locale, coloredlogs

    coloredlogs.install(
        level='DEBUG',
        fmt='[%(levelname)-8s] %(name)-40s %(message)s',
        #stream=open("idtrackerai-gui.log", 'w')
    )
    from pyforms import start_app
    from confapp import conf

    if conf.PYFORMS_MODE=='GUI':
        from .win_idtrackerai  import IdTrackerAiGUI as App
    else:
        from .base_idtrackerai  import BaseIdTrackerAi as App

    start_app( App, geometry=(2800,100,800, 600) )
