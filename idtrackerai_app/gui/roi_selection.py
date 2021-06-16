import cv2
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


def points_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def create_rectangle_points(start, end):
    return [start, (end[0], start[1]), end, (start[0], end[1])]


def create_ellipse_points(start, end):
    width = end[0] - start[0]
    height = end[1] - start[1]
    center = (start[0] + width / 2, start[1] + height / 2)

    distance = points_distance(start, end)
    nPoints = distance / 30
    if nPoints < 8:
        nPoints = 8.0

    points = []
    for angleR in np.arange(0, math.pi * 2, math.pi / nPoints):
        x = int(round(center[0] + width / 2 * np.cos(angleR)))
        y = int(round(center[1] + height / 2 * np.sin(angleR)))
        points.append((x, y))
    return points


def get_intersection_point_distance(test_point, point1, point2):
    p1 = np.float32(point1)
    p2 = np.float32(point2)
    p3 = np.float32(test_point)
    dist = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    return dist


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

    def polybtn_click_evt(self):
        if self._polybtn.checked:
            self._roi += ["[]"]
            self._roi.selected_row_index = len(self._roi) - 1
            self._polybtn.checked = True

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
                    mask, [np.array(points, np.int32)], (255, 255, 255)
                )
        else:
            mask = mask + 255
        return mask

    def roi_selection_changed_evt(self):
        """
        Function called when the selected polygon change
        """
        self._polybtn.checked = False
        self._circlebtn.checked = False

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

    def select_point(self, x, y):
        """
        select a point from the selected polygon

        :param x: X coordenate
        :param y: Y coordenate
        :return: Index of the selected point in the polygon
        """
        index = self._roi.selected_row_index

        # if a polygon is selected
        if index is not None:
            try:
                coord = x, y
                points = eval(self._roi.value[index][0])
                for i, point in enumerate(points):
                    if points_distance(coord, point) <= 5:
                        self._selected_point = i
                        return
                self._selected_point = None
            except:
                pass

    def on_player_double_click_in_video_window(self, event, x, y):
        """
        Event called when the mouse double click.
        This event is used to create vertices on polygons edges.
        :param event: Mouse event
        :param x: X coordenate
        :param y: Y coordenate
        """
        mouse = int(x), int(y)
        distances = []

        index = self._roi.selected_row_index

        # if a polygon is selected
        if index is not None:
            try:
                points = list(eval(self._roi.value[index][0]))
                n_points = len(points)
                for point_index, point in enumerate(points):
                    next_point = points[(point_index + 1) % n_points]
                    distance = get_intersection_point_distance(
                        mouse, point, next_point
                    )

                    if distance <= 5:
                        vector = (
                            next_point[0] - point[0],
                            next_point[1] - point[1],
                        )
                        center = (
                            point[0] + vector[0] / 2,
                            point[1] + vector[1] / 2,
                        )
                        radius = points_distance(center, point)
                        mouse_distance = points_distance(center, mouse)
                        if mouse_distance < radius:
                            distances.append((distance, point_index))
            except:
                pass

            if len(distances) > 0:
                distances = sorted(distances, key=lambda x: x[0])
                point_index = distances[0][1]
                points.insert(point_index + 1, mouse)

                self._roi.set_value(0, index, str(points)[1:-1])

                self._selected_point = point_index + 1

                if not self._player.is_playing:
                    self._player.refresh()

    def on_player_click_in_video_window(self, event, x, y):
        """
        Called when the mouse click in the player
        If the poly button is active it should add the clicked point to the active polygon
        :param event: Mouse event
        :param x: X coordinate
        :param y: Y coordinate
        """
        mouse = int(x), int(y)
        self._selected_point = None

        if (
            not self._rectbtn.checked
            and not self._polybtn.checked
            and not self._circlebtn.checked
        ):
            self.select_point(*mouse)

        if self._polybtn.checked or self._circlebtn.checked:
            index = self._roi.selected_row_index
            if index is not None:
                points = list(eval(self._roi.value[index][0]))
                points.append(mouse)
                if len(points) >= 5 and self._circlebtn.checked:
                    poly = cv2.fitEllipse(np.array(points, np.int32))
                    center, axis, angle = cv2.fitEllipse(
                        np.array(points, np.int32)
                    )
                    center = int(round(center[0])), int(round(center[1]))
                    axis = int(round(axis[0] / 2.0)), int(round(axis[1] / 2.0))
                    angle = int(round(angle))
                    points = cv2.ellipse2Poly(center, axis, angle, 0, 360, 5)
                    self._roi.set_value(
                        0, index, str(points.tolist())[1:-1] + ","
                    )
                    self._circlebtn.checked = False
                else:
                    self._roi.set_value(0, index, str(points)[1:-1] + ",")

    def on_player_drag_in_video_window(self, start_point, end_point):
        """
        Called when the mouse drag start
        :param start_point: Top left point
        :param end_point: Bottom right point
        """

        self._start_point = int(start_point[0]), int(start_point[1])
        self._end_point = int(end_point[0]), int(end_point[1])

        if self._selected_point != None:
            index = self._roi.selected_row_index
            if index is not None:
                try:
                    points = list(eval(self._roi.value[index][0]))
                    points[self._selected_point] = self._end_point
                    self._roi.set_value(0, index, str(points)[1:-1] + ",")
                except Exception as e:
                    print(e)

        # Refresh the image if the video is not playing
        if not self._player.is_playing:
            self._player.refresh()

    def on_player_end_drag_in_video_window(self, start_point, end_point):
        """
        Called when the mouse drag ends
        :param start_point: Top left point
        :param end_point: Bottom right point
        """
        # Draw the rectangle
        if self._rectbtn.checked:

            self._start_point = int(start_point[0]), int(start_point[1])
            self._end_point = int(end_point[0]), int(end_point[1])
            points = create_rectangle_points(
                self._start_point, self._end_point
            )
            self._roi += [str(points)[1:-1]]
            self._start_point = None
            self._end_point = None
            self._rectbtn.checked = False

        # Refresh the image if the video is not playing
        if not self._player.is_playing:
            self._player.refresh()
