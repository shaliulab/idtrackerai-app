from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlNumber
from pyforms.controls import ControlButton


class RangeWin(BaseWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(title="Frames range")

        self._list = kwargs.get("control_list", None)

        self._begin = ControlNumber(
            "Begin",
            default=kwargs.get("begin", 0),
            maximum=kwargs.get("maximum", 100),
        )
        self._end = ControlNumber(
            "End",
            default=kwargs.get("begin", 0),
            maximum=kwargs.get("maximum", 100),
        )
        self._okbtn = ControlButton("Ok", default=self.__okbtn_evt)

        self.formset = [("_begin", "_end"), "_okbtn"]

        self.set_margin(10)

    def __okbtn_evt(self):
        if self._begin.value > self._end.value:
            self.alert(
                "The <b>begin</b> frame has to be smaller than <b>end</b> frame."
            )
        else:
            value = str([int(self._begin.value), int(self._end.value)])
            if self._list.value:
                self._list.value += ", " + value
            else:
                self._list.value = value
            self.close()
