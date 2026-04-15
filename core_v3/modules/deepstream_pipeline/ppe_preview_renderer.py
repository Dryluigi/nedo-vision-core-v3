import time

import pyds


class PPEPreviewRenderer:
    def __init__(
        self,
        frame_drawer,
        class_id_to_label,
        location_name: str,
        person_class_id: int,
        attribute_class_ids,
        preview_style_hold_seconds: float = 3.0,
    ):
        self.frame_drawer = frame_drawer
        self.class_id_to_label = class_id_to_label
        self.location_name = location_name
        self.person_class_id = person_class_id
        self.attribute_class_ids = set(attribute_class_ids)
        self.preview_style_hold_seconds = preview_style_hold_seconds
        self._person_preview_style_cache = {}

    def build_person_style_map(self, persons):
        person_style_by_track_id = {}
        active_person_track_ids = set()

        for person in persons:
            person_id = str(person["person_id"])
            attr_labels = [
                self.class_id_to_label.get(attr["class_id"], str(attr["class_id"]))
                for attr in person.get("attributes", [])
            ]
            active_person_track_ids.add(person_id)
            person_style_by_track_id[person_id] = self._get_stable_person_preview_style(
                person_id,
                attr_labels,
            )

        self._cleanup_preview_style_cache(active_person_track_ids)
        return person_style_by_track_id

    def apply_object_preview(self, batch_meta, frame_meta, obj, frame_w, frame_h, safe_offset, person_style_by_track_id):
        if obj.class_id == self.person_class_id:
            person_track_id = str(obj.object_id)
            person_style = person_style_by_track_id.get(
                person_track_id,
                self._get_stable_person_preview_style(person_track_id, []),
            )
            self._apply_person_preview(
                batch_meta=batch_meta,
                frame_meta=frame_meta,
                obj=obj,
                frame_w=frame_w,
                frame_h=frame_h,
                safe_offset=safe_offset,
                style=person_style,
            )
            return

        if obj.class_id in self.attribute_class_ids:
            self._apply_attribute_preview(
                batch_meta=batch_meta,
                frame_meta=frame_meta,
                obj=obj,
            )

    def _get_person_preview_style(self, labels):
        if any(label in self.frame_drawer.violation_labels for label in labels):
            return {
                "flag": False,
                "line_color": (1.0, 0.2, 0.2, 1.0),
                "border_color": (0.7, 0.7, 0.7, 1.0),
                "gradient_rgb": (0.8, 0.1, 0.1),
                "kind": "violation",
            }
        if labels and all(label in self.frame_drawer.compliance_labels for label in labels):
            return {
                "flag": True,
                "line_color": (0.0, 0.7, 1.0, 1.0),
                "border_color": (0.7, 0.7, 0.7, 1.0),
                "gradient_rgb": (0.0, 0.7, 1.0),
                "kind": "compliance",
            }
        return {
            "flag": None,
            "line_color": (1.0, 1.0, 1.0, 1.0),
            "border_color": (0.7, 0.7, 0.7, 1.0),
            "gradient_rgb": (0.55, 0.55, 0.55),
            "kind": "neutral",
        }

    def _get_stable_person_preview_style(self, track_id, labels):
        now = time.time()
        track_id = str(track_id)
        current_style = self._get_person_preview_style(labels)
        cached = self._person_preview_style_cache.get(track_id)

        if current_style["kind"] != "neutral":
            self._person_preview_style_cache[track_id] = {
                "style": current_style,
                "last_seen_at": now,
            }
            return current_style

        if cached:
            cached["last_seen_at"] = now
            return cached["style"]

        self._person_preview_style_cache[track_id] = {
            "style": current_style,
            "last_seen_at": now,
        }
        return current_style

    def _cleanup_preview_style_cache(self, active_track_ids):
        now = time.time()
        for track_id in list(self._person_preview_style_cache.keys()):
            cached = self._person_preview_style_cache[track_id]
            if track_id not in active_track_ids and (now - cached["last_seen_at"]) > self.preview_style_hold_seconds:
                self._person_preview_style_cache.pop(track_id, None)

    def _get_attribute_preview_style(self, label):
        if label in self.frame_drawer.violation_labels:
            return (1.0, 0.2, 0.2, 1.0)
        if label in self.frame_drawer.compliance_labels:
            return (0.0, 0.7, 1.0, 1.0)
        return (0.7, 0.7, 0.7, 1.0)

    def _apply_person_preview(self, batch_meta, frame_meta, obj, frame_w, frame_h, safe_offset, style):
        obj.text_params.display_text = ""
        obj.text_params.set_bg_clr = 0
        obj.text_params.font_params.font_size = 0

        r = obj.rect_params
        x = int(r.left)
        y = int(r.top)
        w = int(r.width)
        h = int(r.height)

        r.border_width = 1
        r.border_color.set(*style["border_color"])

        corner_len = min(int(w * 0.15), 25)
        thickness = 4

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_lines = 8
        lines = display_meta.line_params

        for i in range(8):
            lines[i].line_width = thickness
            lines[i].line_color.set(*style["line_color"])

        lines[0].x1, lines[0].y1 = x, y
        lines[0].x2, lines[0].y2 = x + corner_len, y
        lines[1].x1, lines[1].y1 = x, y
        lines[1].x2, lines[1].y2 = x, y + corner_len
        lines[2].x1, lines[2].y1 = x + w, y
        lines[2].x2, lines[2].y2 = x + w - corner_len, y
        lines[3].x1, lines[3].y1 = x + w, y
        lines[3].x2, lines[3].y2 = x + w, y + corner_len
        lines[4].x1, lines[4].y1 = x, y + h
        lines[4].x2, lines[4].y2 = x + corner_len, y + h
        lines[5].x1, lines[5].y1 = x, y + h
        lines[5].x2, lines[5].y2 = x, y + h - corner_len
        lines[6].x1, lines[6].y1 = x + w, y + h
        lines[6].x2, lines[6].y2 = x + w - corner_len, y + h
        lines[7].x1, lines[7].y1 = x + w, y + h
        lines[7].x2, lines[7].y2 = x + w, y + h - corner_len
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        gradient_height = 50
        steps = 32
        slice_height = max(1, gradient_height // steps)
        base_r, base_g, base_b = style["gradient_rgb"]
        max_rects = 16
        remaining = steps
        current_step = 0

        while remaining > 0:
            batch_count = min(remaining, max_rects)
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_rects = batch_count

            for i in range(batch_count):
                rect = display_meta.rect_params[i]
                global_index = current_step + i
                rect.left = x
                rect.top = y + h - ((global_index + 1) * slice_height)
                rect.width = w
                rect.height = slice_height + 1
                rect.border_width = 0
                rect.has_bg_color = 1
                progress = global_index / steps
                alpha = 0.2 * (1 - progress**2)
                rect.bg_color.set(base_r, base_g, base_b, alpha)

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            remaining -= batch_count
            current_step += batch_count

        tracker_id = obj.object_id
        confidence_text = f"{obj.confidence:.2f}"

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 3
        texts = display_meta.text_params

        base_x = x + 8
        base_y = y + h - 16
        line_spacing = 12

        texts[0].display_text = f"{tracker_id}"
        texts[0].x_offset = safe_offset(base_x - 1, frame_w - 1)
        texts[0].y_offset = safe_offset(base_y - line_spacing - 4, frame_h - 1)
        texts[0].font_params.font_name = "Serif"
        texts[0].font_params.font_size = 8
        texts[0].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        texts[0].set_bg_clr = 0

        texts[1].display_text = self.location_name or "Unknown"
        texts[1].x_offset = safe_offset(base_x, frame_w - 1)
        texts[1].y_offset = safe_offset(base_y, frame_h - 1)
        texts[1].font_params.font_name = "Serif"
        texts[1].font_params.font_size = 6
        texts[1].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        texts[1].set_bg_clr = 0

        approx_text_width = len(confidence_text) * 10
        texts[2].display_text = confidence_text
        texts[2].x_offset = safe_offset(x + w - approx_text_width + 8, frame_w - 1)
        texts[2].y_offset = safe_offset(base_y, frame_h - 1)
        texts[2].font_params.font_name = "Serif"
        texts[2].font_params.font_size = 6
        texts[2].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        texts[2].set_bg_clr = 0
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    def _apply_attribute_preview(self, batch_meta, frame_meta, obj):
        obj.text_params.display_text = ""
        obj.text_params.set_bg_clr = 0
        obj.text_params.font_params.font_size = 0

        rect = obj.rect_params
        rect.border_width = 2
        label = self.class_id_to_label.get(obj.class_id, str(obj.class_id))
        rect.border_color.set(*self._get_attribute_preview_style(label))
