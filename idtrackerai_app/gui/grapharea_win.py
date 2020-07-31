from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlMatplotlib


class GraphAreaWin(BaseWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(title="Area graph")

        self._graph = ControlMatplotlib(
            "Blobs area", toolbar=False, enabled=False
        )

        self.formset = ["_graph"]

    @property
    def enabled(self):
        return self._graph.enabled

    @enabled.setter
    def enabled(self, value):
        self._graph.enabled = value

    @property
    def on_draw(self):
        return self._graph.on_draw

    @on_draw.setter
    def on_draw(self, value):
        self._graph.on_draw = value

    def draw(self):
        self._graph.draw()
