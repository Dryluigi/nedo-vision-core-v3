from typing import List, Tuple
import cv2
import logging
import os
from pathlib import Path
import threading

import numpy as np
from .DrawingUtils import DrawingUtils

class FrameDrawer:
    """Handles frame processing by drawing objects, annotations, and managing icons."""
    
    # Share the same lock as DrawingUtils for consistent locking
    _cv_lock = DrawingUtils._cv_lock

    def __init__(self):
        self.icons = {}
        self.violation_labels = []
        self.compliance_labels = []
        self.polygons: List[Tuple[Tuple[int, int, int], List[Tuple[float, float]]]] = []
        self.trails = {}
        self.location_name = "LOCATION"
        self.assets_base_path = Path(__file__).resolve().parents[3]

    def update_config(self, icons = {}, violation_labels = [], compliance_labels = []):
        self.icons = self._load_icons(icons)
        self.violation_labels = violation_labels
        self.compliance_labels = compliance_labels

    def _load_icons(self, icon_paths, size=(20, 20)):
        icons = {}
        
        for key, path in icon_paths.items():
            try:
                full_path = Path(path)
                if not full_path.is_absolute():
                    full_path = self.assets_base_path / full_path
                icon = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
                if icon is None:
                    raise FileNotFoundError(f"Icon '{path}' not found at {full_path}.")
                icons[key] = cv2.resize(icon, size)
            except Exception as e:
                logging.warning(f"⚠️ Failed to load icon {path}: {e}")
                icons[key] = None
        return icons

    def _get_color(self, label):
        if label in self.violation_labels:
            return ((0, 0, 255), False)
        elif label in self.compliance_labels:
            return ((255, 141, 11), True)
        else:
            return ((255, 255, 255), None)

    def _get_color_from_labels(self, labels):
        compliance_count = 0
        for label in labels:
            if label in self.violation_labels:
                return ((0, 0, 255), False)
            elif label in self.compliance_labels:
                compliance_count += 1

        if labels and compliance_count == len(labels):
            return ((255, 141, 11), True)
        else:
            return ((255, 255, 255), None)

    def draw_polygons(self, frame):
        height, width = frame.shape[:2]
        for color, normalized_points in self.polygons:
            points = [
                (int(x * width), int(y * height)) for (x, y) in normalized_points
            ]
            if len(points) >= 3:
                with self._cv_lock:
                    cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=color, thickness=2)

    def draw_frame(self, frame, tracked_objects, with_trails=False, trail_length=10):
        current_ids = set()

        for obj in tracked_objects:
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue

            track_id = obj.get('track_id', -1)
            x1, y1, x2, y2 = map(int, bbox)

            attributes = obj.get("attributes", [])
            labels = [attr.get("label") for attr in attributes]
            (color, flag) = self._get_color_from_labels(labels)
            
            with self._cv_lock:
                DrawingUtils.draw_bbox_info(frame, bbox, (color, flag), f"{track_id}", self.location_name, f"{obj.get('confidence', 0):.2f}")

            # Trailing
            if with_trails:
                if not hasattr(self, "trails"):
                    self.trails = {}

                if track_id not in self.trails:
                    self.trails[track_id] = {
                        "points": [],
                        "missed_frames": 0
                    }

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                self.trails[track_id]["points"].append((center_x, center_y))

                # Limit the number of points
                if len(self.trails[track_id]["points"]) > trail_length:
                    self.trails[track_id]["points"] = self.trails[track_id]["points"][-trail_length:]

                points = self.trails[track_id]["points"]
                num_points = len(points)

                # Draw faded trail
                with self._cv_lock:
                    for i in range(1, num_points):
                        cv2.line(frame, points[i-1], points[i], color, 1)

                current_ids.add(track_id)

            with self._cv_lock:
                DrawingUtils.draw_main_bbox(frame, bbox, (color, flag))

                for attr in attributes:
                    attr_bbox = attr.get("bbox", [])         
                    if len(attr_bbox) != 4:
                        continue

                    attr_label = attr.get("label", "")
                    DrawingUtils.draw_inner_box(frame, attr_bbox, self._get_color(attr_label))

                icon_x = x1
                for (label, icon) in self.icons.items():
                    if label in labels:
                        DrawingUtils.draw_alpha_overlay(frame, icon, icon_x, y1 - 25)
                        icon_x += 25

        # Cleanup trails for objects that disappeared
        if with_trails and hasattr(self, "trails"):
            for track_id in list(self.trails.keys()):
                if track_id not in current_ids:
                    self.trails[track_id]["missed_frames"] += 1
                else:
                    self.trails[track_id]["missed_frames"] = 0

                if self.trails[track_id]["missed_frames"] > 30:
                    del self.trails[track_id]

        return frame
