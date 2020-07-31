import cv2
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


def points_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def create_rectangle_points(start, end):
    return [start, (end[0], start[1]), end, (start[0], end[1])]


def get_intersection_point_distance(test_point, point1, point2):
    p1 = np.float32(point1)
    p2 = np.float32(point2)
    p3 = np.float32(test_point)
    dist = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    return dist


class PlayerWinInteractions(object):
    def __init__(self):
        self._selected_point = None  # point used to edit the polygon vertice
        self._start_point = None  # square starting point
        self._end_point = None  # square ending point

    def select_point(self, x, y):
        """
        select a point from the selected polygon

        :param x: X coordenate
        :param y: Y coordenate
        :return: Index of the selected point in the polygon
        """
        if (
            self._polybtn.checked
            or self._circlebtn.checked
            or self._rectbtn.checked
        ):
            points_list = self._roi
        else:
            points_list = self._points_list
        index = points_list.selected_row_index

        if index is not None:
            try:
                coord = x, y
                points = eval(points_list.value[index][0])
                for i, point in enumerate(points):
                    if points_distance(coord, point) <= 5:
                        self._selected_point = i
                        return
                self._selected_point = None
            except Exception as e:
                print(e)
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
            ## TODO: Do not use bare except
            except Exception as e:
                print(e)
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
            and not self._add_points_btn.checked
        ):
            self.select_point(*mouse)

        if (
            self._polybtn.checked
            or self._circlebtn.checked
            or self._add_points_btn.checked
        ):
            if self._polybtn.checked or self._circlebtn.checked:
                points_list = self._roi
            else:
                points_list = self._points_list
            index = points_list.selected_row_index

            if index is not None:
                points = list(eval(points_list.value[index][0]))
                points.append(mouse)
                if len(points) >= 5 and self._circlebtn.checked:
                    center, axis, angle = cv2.fitEllipse(
                        np.array(points, np.int32)
                    )
                    center = int(round(center[0])), int(round(center[1]))
                    axis = int(round(axis[0] / 2.0)), int(round(axis[1] / 2.0))
                    angle = int(round(angle))
                    points = cv2.ellipse2Poly(center, axis, angle, 0, 360, 5)
                    points_list.set_value(
                        0, index, str(points.tolist())[1:-1] + ","
                    )
                    self._circlebtn.checked = False
                else:
                    points_list.set_value(0, index, str(points)[1:-1] + ",")

    def on_player_drag_in_video_window(self, start_point, end_point):
        """
        Called when the mouse drag start
        :param start_point: Top left point
        :param end_point: Bottom right point
        """

        self._start_point = int(start_point[0]), int(start_point[1])
        self._end_point = int(end_point[0]), int(end_point[1])

        if self._selected_point is not None:
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
