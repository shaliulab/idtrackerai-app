import math
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)


def points_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


class SetupInfoWin(object):
    def __init__(self):
        self._selected_point = None  # point used to edit the polygon vertice

    def add_points_dict_click_evt(self):
        # Initialize list
        if self._add_points_btn.checked:
            name = self.input_text(
                "Input name of list of points to be added",
                title="List of points name",
                default=None,
            )
            self._points_list += ['["{}", ]'.format(name)]
            self._points_list.selected_row_index = len(self._points_list) - 1
            self._add_points_btn.checked = True
            self._polybtn.checked = False
            self._circlebtn.checked = False
            self._rectbtn.checked = False

    def add_setup_info_changed_evt(self):
        """
        Function called when the selected polygon change
        """
        self._add_points_btn.checked = False
        self._polybtn.checked = False
        self._circlebtn.checked = False
        self._rectbtn.checked = False

    def remove_setup_info(self):
        self._points_list -= -1
        self._points_list.selected_row_index = None  # No poly is selected
        self._player.refresh()

    def draw_points_list(self, frame):
        """
        Draw the ROIs lines in the frame.
        """
        # Draw selected points
        try:
            index = self._points_list.selected_row_index

            for row_index, row in enumerate(self._points_list.value):
                points = eval(row[0])
                # draw vertices
                for i, point in enumerate(points):
                    if not isinstance(point, str):
                        point = tuple(point)
                        if self._selected_point == i and index == row_index:
                            cv2.circle(frame, point, 4, (0, 0, 255), 2)
                        else:
                            cv2.circle(frame, point, 4, (0, 255, 0), 2)

        except Exception as e:
            logger.debug(str(e), exc_info=True)

        return frame

    def remove_points_list(self):
        self._points_list -= -1
        self._points_list.selected_row_index = None  # No poly is selected
        self._player.refresh()

    def create_setup_poitns_dict(self):
        if len(self._points_list) > 0:
            points_dict = {}
            for row in self._points_list.value:

                if isinstance(row, str):
                    points = eval(row)
                elif isinstance(row[0], str):
                    points = eval(row[0])
                else:
                    points = row

                points_dict[points[0]] = np.asarray(points[1:])

            return points_dict
        else:
            return None
