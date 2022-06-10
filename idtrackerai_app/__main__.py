import sys, os

sys.path.append(os.getcwd())


def start():
    import logging
    from rich.logging import RichHandler
    from rich.console import Console

    logger_width_when_no_terminal = 150
    try:
        os.get_terminal_size()
    except OSError:
        # stdout is sent to file. We define logger width to a constant
        size = logger_width_when_no_terminal
    else:
        # stdout is sent to terminal
        # We define logger width to adapt to the terminal width
        size = None

    # The first handler is the terminal, the second one the .log file,
    # both rendered with Rich and full logging (level=0)
    logging.basicConfig(
        level=0,
        format="%(message)s",
        datefmt="%b %d %H:%M:%S",
        handlers=[
            RichHandler(console=Console(width=size)),
            RichHandler(
                console=Console(
                    file=open("idtrackerai-app.log", "w"),
                    width=logger_width_when_no_terminal,
                ),
            ),
        ],
    )

    logger = logging.getLogger()
    from pyforms import start_app
    from confapp import conf

    logging.getLogger("PyQt5").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    if conf.PYFORMS_MODE == "GUI":
        from .win_idtrackerai import IdTrackerAiGUI as App
    else:
        from .base_idtrackerai import BaseIdTrackerAi as App

    try:
        start_app(App, geometry=(100, 100, 800, 600))
    except SystemExit:
        pass
    except Exception as e:
        logger.info(e, exc_info=True)
        import traceback

        ex_type, ex, tb = sys.exc_info()
        traceback.print_exception(ex_type, ex, tb)


# Execute the application
if __name__ == "__main__":
    start()
