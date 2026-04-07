import math
import cv2
import numpy as np
import os
import threading

class DrawingUtils:
    # Global lock for all OpenCV drawing operations to prevent segfaults
    _cv_lock = threading.Lock()
    
    _color_map = {
        True: "blue",
        False: "red",
        None: "blue"
    }


    _corner_images_by_color = {}
    _line_images_by_color = {}
    _inner_frame_image_by_color = {}
    _corner_half_by_color = {}
    _line_half_by_color = {}
    _line_size_by_color = {}
    _corner_quarter_by_color = {}
    
    _asset_cache = {}
    _base_height = 720
    _scale_weight = 1.3
    _max_scale = 3.0
    
    @staticmethod
    def initialize(assets_path: str):
        for color_flag, color_name in DrawingUtils._color_map.items():
            top_left = cv2.imread(os.path.join(assets_path, color_name, "top_left.png"), cv2.IMREAD_UNCHANGED)
            top_right = cv2.imread(os.path.join(assets_path, color_name, "top_right.png"), cv2.IMREAD_UNCHANGED)

            DrawingUtils._corner_images_by_color[color_flag] = {
                'top_left': top_left,
                'top_right': top_right,
                'bottom_left': cv2.flip(top_right, -1),
                'bottom_right': cv2.flip(top_left, -1)
            }

            line = cv2.imread(os.path.join(assets_path, color_name, "line.png"), cv2.IMREAD_UNCHANGED)
            DrawingUtils._line_images_by_color[color_flag] = {
                'vertical': line,
                'horizontal': cv2.rotate(line, cv2.ROTATE_90_CLOCKWISE)
            }
            
            inner_frame = cv2.imread(os.path.join(assets_path, color_name, "inner_frame.png"), cv2.IMREAD_UNCHANGED)
            DrawingUtils._inner_frame_image_by_color[color_flag] = inner_frame
            
            # Initialize cache for the base resolution
            DrawingUtils._prepare_scaled_assets(DrawingUtils._base_height, color_flag)

    @staticmethod
    def _get_scale_factor(height):
        if height <= DrawingUtils._base_height:
            return height / DrawingUtils._base_height * DrawingUtils._scale_weight
        
        base_scale = DrawingUtils._scale_weight
        height_ratio = height / DrawingUtils._base_height
        additional_scale = (math.sqrt(height_ratio) - 1) * 0.5
        
        return min(base_scale + additional_scale, DrawingUtils._max_scale)

    @staticmethod
    def _prepare_scaled_assets(frame_height, color_flag):
        """Prepare and cache scaled assets for a specific frame height"""
        scale_factor = DrawingUtils._get_scale_factor(frame_height)
        key = DrawingUtils._color_map.get(color_flag)
        
        if (frame_height, key) in DrawingUtils._asset_cache:
            return DrawingUtils._asset_cache[(frame_height, key)]
            
        corner_size = int(80 * scale_factor)
        
        scaled_corners = {}
        for position, img in DrawingUtils._corner_images_by_color[color_flag].items():
            scaled_corners[position] = cv2.resize(img, (corner_size, corner_size))
            
        line_size = int(15 * scale_factor)
        
        scaled_lines = {}
        for direction, img in DrawingUtils._line_images_by_color[color_flag].items():
            scaled_lines[direction] = cv2.resize(img, (line_size, line_size))
        
        corner_half = corner_size // 2
        corner_quarter = corner_size // 4
        line_half = line_size // 2
        
        cache_entry = {
            'corners': scaled_corners,
            'lines': scaled_lines,
            'line_size': line_size,
            'line_half': line_half,
            'corner_half': corner_half,
            'corner_quarter': corner_quarter,
            'scale_factor': scale_factor
        }
        
        DrawingUtils._asset_cache[(frame_height, key)] = cache_entry
        return cache_entry

    @staticmethod
    def draw_alpha_overlay(dest, src, x, y):
        if src is None or dest is None:
            return

        # Get source dimensions
        src_h, src_w = src.shape[:2]
        
        # Calculate safe region bounds
        dest_h, dest_w = dest.shape[:2]
        y_start = max(y, 0)
        x_start = max(x, 0)
        y_end = min(y + src_h, dest_h)
        x_end = min(x + src_w, dest_w)
        
        # Calculate source crop coordinates
        crop_y1 = max(-y, 0)
        crop_x1 = max(-x, 0)
        crop_y2 = src_h - max((y + src_h) - dest_h, 0)
        crop_x2 = src_w - max((x + src_w) - dest_w, 0)

        # Check if there's any valid area to draw
        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return

        # Crop source image to valid region
        src_cropped = src[crop_y1:crop_y2, crop_x1:crop_x2]
        roi = dest[y_start:y_end, x_start:x_end]

        if src_cropped.shape[2] == 4:
            # Split source into color and alpha channels
            src_bgr = src_cropped[:, :, :3]
            alpha = src_cropped[:, :, 3:] / 255.0
            
            # Blend with ROI using alpha
            if roi.shape[:2] == src_bgr.shape[:2]:
                blended = (src_bgr * alpha) + (roi[:, :, :3] * (1 - alpha))
                roi[:, :, :3] = blended.astype(np.uint8)
        else:
            if roi.shape == src_cropped.shape:
                roi[:] = src_cropped

    @staticmethod
    def draw_bbox_info(frame, bbox, color_data, title, subtitle, suffix):
        color, flag = color_data
        x1, y1, x2, y2 = map(int, bbox)
        
        frame_height = frame.shape[0]
        scale_factor = DrawingUtils._get_scale_factor(frame_height)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25 * scale_factor
        title_scale = 0.4 * scale_factor

        # Colors
        text_color = (255, 255, 255)

        # Measure text sizes
        (_, title_h), _ = cv2.getTextSize(title, font, title_scale, 1)
        (_, subtitle_h), _ = cv2.getTextSize(subtitle, font, font_scale, 1)
        (suffix_w, _), _ = cv2.getTextSize(suffix, font, font_scale, 1)

        padding = int(15 * scale_factor)
        line_spacing = int(7 * scale_factor)

        subtitle_y = y2 - padding
        title_y = subtitle_y - subtitle_h - line_spacing
        block_height = title_h + subtitle_h + line_spacing + 2 * padding

        alpha_start = 0.6
        alpha_end = 0

        for i in range(block_height):
            relative_pos = i / block_height
            alpha = alpha_start + (alpha_end - alpha_start) * (1 - relative_pos) 
            
            y_pos = y2 - block_height + i
            if y_pos < 0 or y_pos >= frame.shape[0]:
                continue

            original_row = frame[y_pos, x1:x2].astype(np.float32)
            blended_row = (1 - alpha) * original_row + alpha * np.array(color, dtype=np.float32)
            frame[y_pos, x1:x2] = blended_row.astype(np.uint8)

        # Draw texts
        cv2.putText(frame, title, (x1 + padding, title_y), font, title_scale, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (x1 + padding, subtitle_y), font, font_scale, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, suffix, (x2 - suffix_w - padding, subtitle_y), font, font_scale, text_color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_corner_line(frame, bbox, color, thickness=1):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Scale the corner length based on frame height
        frame_height = frame.shape[0]
        scale_factor = DrawingUtils._get_scale_factor(frame_height)
        
        # Scale thickness
        scaled_thickness = max(1, int(thickness * scale_factor))
        
        corner_length = min(x2 - x1, y2 - y1) // 6

        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, scaled_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, scaled_thickness)

        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, scaled_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, scaled_thickness)

        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, scaled_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, scaled_thickness)

        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, scaled_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, scaled_thickness)

    @staticmethod
    def draw_main_bbox(frame, bbox, color_data):
        color, flag = color_data

        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        
        frame_height = frame.shape[0]
        
        assets_data = DrawingUtils._prepare_scaled_assets(frame_height, flag)
        if not assets_data:
            return frame
            
        assets = assets_data['corners']
        lines = assets_data['lines']
        line_size = assets_data['line_size']
        half_line = assets_data['line_half']
        half_corner = assets_data['corner_half']
        quarter_corner = assets_data['corner_quarter']
        scale_factor = assets_data['scale_factor']
        
        # Define minimum size threshold for corners (scaled)
        min_size_threshold = int(half_corner * scale_factor)
        
        if h < min_size_threshold or w < min_size_threshold:
            half_corner = 0
            quarter_corner = 0

        # Draw vertical lines (optimized)
        vertical_line = lines['vertical']
        if vertical_line is not None:
            line_h = h - half_corner
            if line_h > 0:
                line = cv2.resize(vertical_line, (line_size, line_h))
                DrawingUtils.draw_alpha_overlay(frame, line, x1 - half_line, y1 + quarter_corner)
                DrawingUtils.draw_alpha_overlay(frame, line, x2 - half_line, y1 + quarter_corner)

        # Draw horizontal lines (optimized)
        horizontal_line = lines['horizontal']
        if horizontal_line is not None:
            line_w = w - half_corner
            if line_w > 0:
                line = cv2.resize(horizontal_line, (line_w, line_size))
                DrawingUtils.draw_alpha_overlay(frame, line, x1 + quarter_corner, y1 - half_line)
                DrawingUtils.draw_alpha_overlay(frame, line, x1 + quarter_corner, y2 - half_line)

        if h < min_size_threshold or w < min_size_threshold:
            # For small bounding boxes, just use lines
            scaled_thickness = max(1, int(2 * scale_factor))
            DrawingUtils.draw_corner_line(frame, bbox, color, scaled_thickness)
        else:
            # Draw corners
            DrawingUtils.draw_alpha_overlay(frame, assets['top_left'], x1 - half_corner, y1 - half_corner)
            DrawingUtils.draw_alpha_overlay(frame, assets['top_right'], x2 - half_corner, y1 - half_corner)
            DrawingUtils.draw_alpha_overlay(frame, assets['bottom_left'], x1 - half_corner, y2 - half_corner)
            DrawingUtils.draw_alpha_overlay(frame, assets['bottom_right'], x2 - half_corner, y2 - half_corner)

        return frame

    @staticmethod
    def draw_inner_box(frame, bbox, color_data, thickness=1):
        color, flag = color_data

        if flag is None:
            return frame

        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        
        frame_height = frame.shape[0]
        scale_factor = DrawingUtils._get_scale_factor(frame_height)
        
        # Scale thickness
        scaled_thickness = max(1, int(thickness * scale_factor))
        
        # Get original texture
        texture = DrawingUtils._inner_frame_image_by_color.get(flag)

        if texture is None:
            return frame

        # Draw texture with alpha blending
        if texture.shape[2] == 4:
            resized_texture = cv2.resize(texture, (w, h))
            DrawingUtils.draw_alpha_overlay(frame, resized_texture, x1, y1)
        else:
            frame[y1:y2, x1:x2] = cv2.resize(texture, (w, h))

        DrawingUtils.draw_corner_line(frame, bbox, color, scaled_thickness)

        return frame
    
    @staticmethod
    def crop_with_bounding_box(frame, obj, target_height=512, buffer=30):
        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, obj["bbox"])

        current_height = y2 - y1
        target_height = target_height if target_height else current_height + buffer * 2
        scale = (target_height - buffer * 2) / current_height

        # Calculate buffer based on edge proximity
        scaled_buffer = int(buffer / scale)
        
        # Check if bounding box is at the edges
        at_left_edge = x1 <= scaled_buffer
        at_right_edge = x2 >= img_w - scaled_buffer
        at_top_edge = y1 <= scaled_buffer
        at_bottom_edge = y2 >= img_h - scaled_buffer
        
        # Apply buffer only if not at edges
        crop_x1 = max(x1 - (0 if at_left_edge else scaled_buffer), 0)
        crop_y1 = max(y1 - (0 if at_top_edge else scaled_buffer), 0)
        crop_x2 = min(x2 + (0 if at_right_edge else scaled_buffer), img_w)
        crop_y2 = min(y2 + (0 if at_bottom_edge else scaled_buffer), img_h)

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        _, cropped_w = cropped.shape[:2]

        target_width = int(cropped_w * scale)
        final_img = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)

        def transform_bbox(bbox):
            x1, y1, x2, y2 = bbox

            nx1 = int((x1 - crop_x1) * scale)
            ny1 = int((y1 - crop_y1) * scale)
            nx2 = int((x2 - crop_x1) * scale)
            ny2 = int((y2 - crop_y1) * scale)
            return (nx1, ny1, nx2, ny2)

        obj = obj.copy()
        obj["bbox"] = transform_bbox(obj["bbox"])

        for attr in obj.get("attributes", []):
            if "bbox" in attr:
                attr["bbox"] = transform_bbox(attr["bbox"])

        return final_img, obj

