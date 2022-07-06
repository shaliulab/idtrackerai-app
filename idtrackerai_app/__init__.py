__version__ = "1.0.8"

import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")

# TODO
# Is this actually needed?
# Or is there a code in pyforms that always imports the local_settings?
from confapp import conf

try:
    import local_settings
    conf += local_settings
except:
    pass
