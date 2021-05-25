import cv2
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ROISelectionWin(object):
    def __init__(self):
        self._selected_point = None  # point used to edit the polygon vertice
        self._start_point = None  # square starting point
        self._end_point = None  # square ending point

    def circlebtn_click_evt(self):
        if self._circlebtn.checked:
            self._roi += ["[]"]
            self._roi.selected_row_index = len(self._roi) - 1
            self._circlebtn.checked = True
            self._polybtn.checked = False
            self._rectbtn.checked = False
            self._add_points_btn.checked = False

    def polybtn_click_evt(self):
        if self._polybtn.checked:
            self._roi += ["[]"]
            self._roi.selected_row_index = len(self._roi) - 1
            self._polybtn.checked = True
            self._circlebtn.checked = False
            self._rectbtn.checked = False
            self._add_points_btn.checked = False

    def rectbtn_click_evt(self):
        if self._rectbtn.checked:
            self._polybtn.checked = False
            self._circlebtn.checked = False
            self._rectbtn.checked = True
            self._add_points_btn.checked = False

    def create_mask(self, height, width):
        """
        Create a mask based on the selected ROIs
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        if len(self._roi) > 0:
            for row in self._roi.value:
                if isinstance(row, str):
                    points = eval(row)
                elif isinstance(row[0], str):
                    points = eval(row[0])
                else:
                    points = row
                if len(points) < 3:
                    continue
                mask = cv2.fillPoly(
                    mask, [np.array(points, np.int32)], (1, 1, 1)
                )
        else:
            mask = mask + 1
        return mask

    def roi_selection_changed_evt(self):
        """
        Function called when the selected polygon change
        """
        self._polybtn.checked = False
        self._circlebtn.checked = False
        self._rectbtn.checked = False
        self._add_points_btn.checked = False

    def remove_roi(self):
        self._roi -= -1
        self._roi.selected_row_index = None  # No poly is selected
        self._player.refresh()

    def draw_rois(self, frame):
        """
        Draw the ROIs lines in the frame.
        """
        # Draw polygons
        try:
            index = self._roi.selected_row_index

            for row_index, row in enumerate(self._roi.value):
                points = eval(row[0])

                if len(points) > 3:
                    cv2.polylines(
                        frame,
                        [np.array(points, np.int32)],
                        True,
                        (0, 255, 0),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                else:
                    # draw vertices
                    for i, point in enumerate(points):
                        point = tuple(point)
                        if self._selected_point == i and index == row_index:
                            cv2.circle(frame, point, 4, (0, 0, 255), 2)
                        else:
                            cv2.circle(frame, point, 4, (0, 255, 0), 2)

                if (
                    index == row_index
                    and self._selected_point is not None
                    and len(points) > self._selected_point
                ):
                    point = tuple(points[self._selected_point])
                    cv2.circle(frame, point, 4, (0, 0, 255), 2)

        except Exception as e:
            logger.debug(str(e), exc_info=True)

        # Draw the polygons in edition
        if self._start_point and self._end_point:

            if self._rectbtn.checked:
                cv2.rectangle(
                    frame, self._start_point, self._end_point, (233, 44, 44), 1
                )

        return frame
