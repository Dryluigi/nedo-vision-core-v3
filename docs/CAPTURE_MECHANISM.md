# Nedo Vision Worker Core V2 Inspection Handoff

Prepared from local code inspection on 2026-04-07. This document summarizes the runtime flow, capture mechanism, storage and database persistence, and drawing implementation so it can be reused in another project context.

## 1. High-Level Runtime Flow

The application starts in `CoreService.initialize()`, which sets environment variables, initializes local SQLite databases, initializes drawing assets, and starts two background sync threads: one for video sources and one for AI pipelines.

Relevant files:

- `nedo_vision_worker_core/core_service.py`
- `nedo_vision_worker_core/database/DatabaseManager.py`
- `nedo_vision_worker_core/streams/StreamSyncThread.py`
- `nedo_vision_worker_core/pipeline/PipelineSyncThread.py`

Rough flow:

```text
CoreService
  -> DatabaseManager.init_databases(storage_path)
  -> StreamSyncThread polls worker_source table
  -> PipelineSyncThread polls worker_source_pipeline table
  -> PipelineManager starts a PipelineProcessor
  -> PipelineProcessor reads frames from VideoStreamManager
  -> detection thread runs detection + tracking
  -> repositories save images and database rows when trigger conditions are met
```

## 2. Storage Layout

The storage root comes from `storage_path` and is resolved by `DatabaseManager.init_databases()`.

| Path                                    | Purpose                                                      |
| --------------------------------------- | ------------------------------------------------------------ |
| `<storage>/sqlite/default.db`           | Main runtime SQLite database for detections and runtime data |
| `<storage>/sqlite/config.db`            | Configuration database for sources, pipelines, configs       |
| `<storage>/sqlite/logging.db`           | Logging-related SQLite database                              |
| `<storage>/files/ppe_detections`        | Saved PPE full-frame and cropped images                      |
| `<storage>/files/restricted_violations` | Saved restricted-area full-frame and cropped images          |
| `<storage>/files/detection_image`       | Generic violation image for webhook/MQTT pipeline events     |
| `<storage>/files/debug_image`           | Debug snapshots generated from debug requests                |
| `<storage>/files/source_files`          | Expected local path used for file-based video sources        |

## 3. Capture Mechanism

### 3.1 Source Registration and Stream Start

`StreamSyncThread` polls the `worker_source` table from the config database and registers sources into `VideoStreamManager`. Streams are lazily started: the stream does not begin capturing until a pipeline acquires it.

Relevant files:

- `nedo_vision_worker_core/repositories/WorkerSourceRepository.py`
- `nedo_vision_worker_core/streams/StreamSyncThread.py`
- `nedo_vision_worker_core/streams/VideoStreamManager.py`

### 3.2 Pipeline Start

`PipelineSyncThread` polls the `worker_source_pipeline` table. When a pipeline is in `starting` or `restarting` state, it is started through `PipelineManager`.

Relevant files:

- `nedo_vision_worker_core/repositories/WorkerSourcePipelineRepository.py`
- `nedo_vision_worker_core/pipeline/PipelineSyncThread.py`
- `nedo_vision_worker_core/pipeline/PipelineManager.py`

### 3.3 Frame Acquisition

There are two main capture paths:

- Regular streams/files: `VideoStream` opens RTSP/file input using GStreamer or FFmpeg and continuously publishes the latest frame.
- Direct camera devices: `SharedVideoDeviceManager` routes access through a sharing layer so multiple consumers can use the same device. It may use `VideoSharingDaemon` and a client for cross-process access.

Relevant files:

- `nedo_vision_worker_core/streams/VideoStream.py`
- `nedo_vision_worker_core/streams/SharedVideoDeviceManager.py`
- `nedo_vision_worker_core/services/VideoSharingDaemon.py`

## 4. Violation Capture Mechanism

Violation capture is not triggered on the first frame. The system waits until the same tracked person has a stable matching attribute for `5` consecutive detection cycles.

### 4.0 Detection Output Schema

All detectors in this project return detections in the same shape:

```python
{
  "label": str,
  "confidence": float,
  "bbox": [x1, y1, x2, y2]
}
```

Important details:

- The base contract is defined in `BaseDetector.detect_objects()`.
- `YOLODetector` returns `box.xyxy.tolist()[0]`, so bbox values are absolute image coordinates in `[x1, y1, x2, y2]` format.
- `RFDETRDetector` also returns `xyxy`, again in absolute image coordinates.

Relevant files:

- `nedo_vision_worker_core/detection/BaseDetector.py`
- `nedo_vision_worker_core/detection/YOLODetector.py`
- `nedo_vision_worker_core/detection/RFDETRDetector.py`

### 4.1 Stable Detection Counter

`TrackerManager` assigns a persistent UUID to each tracked person and maintains:

- `track_count_map`: how many times the person has been seen
- `track_attributes_presence`: consecutive presence counts for each attribute label

If an attribute is not present on the current frame, that label’s counter is reset to `0`. This means the threshold is consecutive, not cumulative.

Relevant file:

- `nedo_vision_worker_core/tracker/TrackerManager.py`

### 4.2 PPE Violation Matching

For PPE, detected attribute boxes are matched to person boxes by coverage overlap in `PersonAttributeMatcher`. The result is a per-person object with filtered attributes such as `helmet`, `no_helmet`, `vest`, `no_vest`, and others.

The matching math is implemented in `BoundingBoxMetrics.compute_coverage(box1, box2)`:

```text
coverage = intersection_area(person_bbox, attribute_bbox) / area(attribute_bbox)
```

That means an attribute is considered attached to a person when enough of the attribute box lies inside the person box.

Current threshold used by `PPEDetectionProcessor`:

- `coverage_threshold = 0.5`

So an attribute bbox is matched to a person when at least 50% of the attribute bbox is covered by the person bbox.

After that:

- exclusive pairs like `helmet` vs `no_helmet` are reduced to the highest-confidence one
- multi-instance classes like gloves/boots/goggles may keep multiple boxes
- the result becomes a person object with `attributes`

Relevant files:

- `nedo_vision_worker_core/util/BoundingBoxMetrics.py`
- `nedo_vision_worker_core/util/PersonAttributeMatcher.py`
- `nedo_vision_worker_core/detection/detection_processing/PPEDetectionProcessor.py`

### 4.3 Restricted Area Matching

For restricted-area violations, a person is marked with the attribute `in_restricted_area` when the center point of the person’s bounding box falls inside a configured polygon.

The restricted-area rule is:

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
in_restricted = polygon.contains(Point(center_x, center_y))
```

If true, the person gets:

```python
{
  "label": "in_restricted_area",
  "confidence": 1.0
}
```

Relevant files:

- `nedo_vision_worker_core/util/PersonRestrictedAreaMatcher.py`
- `nedo_vision_worker_core/detection/detection_processing/HumanDetectionProcessor.py`

### 4.4 Save Trigger

The detection worker in `PipelineProcessor` runs tracking results through repository persistence only when feature flags are enabled:

- `db` enables database/image persistence
- `webhook` or `mqtt` enables generic detection snapshot persistence

Relevant files:

- `nedo_vision_worker_core/pipeline/PipelineProcessor.py`
- `nedo_vision_worker_core/pipeline/PipelineConfigManager.py`
- `nedo_vision_worker_core/repositories/WorkerSourcePipelineRepository.py`

### 4.5 Actual Capture Conditions

| Flow                      | Condition                                              | Repository                                                 |
| ------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| PPE detection save        | Any attribute on a tracked person reaches `count == 5` | `PPEDetectionRepository.save_ppe_detection()`              |
| Restricted area save      | `in_restricted_area` reaches `count == 5`              | `RestrictedAreaRepository.save_area_violation()`           |
| Webhook/MQTT generic save | Only saves when a violating label is present           | `WorkerSourcePipelineDetectionRepository.save_detection()` |

Important nuance: the PPE repository is not strictly violation-only. It persists once any tracked PPE attribute reaches the threshold, even compliant labels, while the generic webhook/MQTT path filters specifically for violation labels.

### 4.6 How `count == 5` Is Produced

The `count` used by the repositories is not stored in the DB first. It is produced in memory by `TrackerManager`.

Flow:

1. Detection processor produces a person object with `attributes`
2. `TrackerManager` associates the person to a track ID
3. `TrackerManager` assigns a persistent per-person UUID
4. `TrackerManager._update_attribute_presence()` increments each attribute label count if present on this cycle, or resets it to `0` if absent
5. `TrackerManager._generate_tracking_results()` copies the current per-label counter into each attribute as `attr["count"]`
6. Repository checks that count and decides whether to save

Resulting tracked object schema:

```python
{
  "uuid": str,
  "track_id": int,
  "detections": int,
  "label": "person",
  "confidence": float,
  "bbox": [x1, y1, x2, y2],
  "attributes": [
    {
      "label": str,
      "confidence": float,
      "bbox": [x1, y1, x2, y2],   # optional for area violation
      "count": int
    }
  ]
}
```

This tracked object is the main data structure passed into drawing and repository save methods.

## 5. How Images Are Saved

### 5.1 PPE Save Path

`PPEDetectionRepository.save_ppe_detection()` does the following for each triggered tracked object:

- Builds a filtered object with stable attributes
- Renders a full annotated frame
- Saves the full frame to `<storage>/files/ppe_detections`
- Crops the person region with a buffer
- Renders the cropped person image with adjusted boxes
- Saves the cropped image to the same folder
- Inserts a row into `ppe_detections`
- Inserts child rows into `ppe_detection_labels`

### 5.2 Restricted Area Save Path

`RestrictedAreaRepository.save_area_violation()` performs a similar flow:

- Draws polygons on the frame first
- Renders the full annotated frame for the person
- Saves full frame and cropped image to `<storage>/files/restricted_violations`
- Inserts a row into `restricted_area_violation`

### 5.3 Generic Detection Save Path

`WorkerSourcePipelineDetectionRepository` saves a single annotated image and JSON payload into `worker_source_pipeline_detection` when webhook or MQTT outputs are enabled and a violating label exists.

### 5.4 Exact Database Column Mapping

#### `ppe_detections`

Main image-level record for PPE saves:

- `id`: generated UUID
- `worker_id`: pipeline ID
- `worker_source_id`: source ID
- `person_id`: tracked person UUID
- `image_path`: full annotated frame path
- `image_tile_path`: cropped person image path
- `detection_count`: `tracked_obj["detections"]`
- `created_at`: insert time
- `b_box_x1`, `b_box_y1`, `b_box_x2`, `b_box_y2`: person bbox from `tracked_obj["bbox"]`

#### `ppe_detection_labels`

Child rows for each saved PPE attribute:

- `id`: generated UUID
- `detection_id`: FK to `ppe_detections.id`
- `code`: label such as `helmet`, `no_helmet`, `vest`, `no_vest`
- `confidence_score`: attribute confidence
- `detection_count`: attribute stability counter, from `attr["count"]`
- `b_box_x1`, `b_box_y1`, `b_box_x2`, `b_box_y2`: attribute bbox if available, else `0.0`

#### `restricted_area_violation`

Area violation image-level record:

- `id`: generated UUID
- `worker_source_id`: source ID
- `person_id`: tracked person UUID
- `image_path`: full annotated frame path
- `image_tile_path`: cropped person image path
- `confidence_score`: person confidence
- `created_at`: insert time
- `b_box_x1`, `b_box_y1`, `b_box_x2`, `b_box_y2`: person bbox

#### `worker_source_pipeline_detection`

Generic webhook/MQTT detection snapshot:

- `id`: generated UUID
- `worker_source_pipeline_id`: pipeline ID
- `image_path`: saved full-frame image path
- `data`: JSON-serialized filtered tracked objects
- `created_at`: insert time

#### `worker_source_pipeline_debug`

Debug response table:

- `id`: generated UUID
- `uuid`: external debug request UUID
- `worker_source_pipeline_id`: pipeline ID
- `image_path`: saved debug image path
- `data`: JSON payload of tracked objects
- `created_at`: insert time

## 6. Drawing Mechanism

### 6.1 Main Drawing Classes

| Class          | Responsibility                                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `FrameDrawer`  | Top-level renderer for full object drawing, icons, inner boxes, polygons, and trails                              |
| `DrawingUtils` | Low-level drawing helper for bbox frames, text panels, alpha overlay, inner boxes, and crop coordinate transforms |

### 6.2 How the Bounding Box Is Drawn

The repositories do not draw boxes directly. Instead, they call `frame_drawer.draw_frame(...)` before saving images with `cv2.imwrite(...)`.

`FrameDrawer.draw_frame()` does this:

- Reads the person bbox from `obj["bbox"]`
- Chooses style/color from attribute labels
- Draws the text/info strip using `DrawingUtils.draw_bbox_info()`
- Draws the outer bbox using `DrawingUtils.draw_main_bbox()`
- Draws attribute sub-boxes using `DrawingUtils.draw_inner_box()`
- Draws icons above the box if configured

Exact drawing data flow:

1. Repository receives `tracked_obj`
2. Repository prepares object for drawing
   PPE path may filter attributes first to only those with `count >= 5`
3. Repository calls `frame_drawer.draw_frame(frame.copy(), [obj])`
4. `FrameDrawer` extracts `bbox`, `track_id`, `confidence`, and `attributes`
5. `FrameDrawer` computes box color by looking at labels:
   - violation label present -> red style
   - all labels compliance -> blue/orange style
   - otherwise neutral/white fallback
6. `DrawingUtils.draw_bbox_info()` draws the bottom info strip and text
7. `DrawingUtils.draw_main_bbox()` draws the stylized outer person box
8. For each attribute with its own bbox, `DrawingUtils.draw_inner_box()` draws the smaller attribute box
9. `FrameDrawer` overlays icons for matched labels
10. Repository writes the rendered image to disk with `cv2.imwrite(...)`

### 6.3 Outer Box Styling

The outer box is not a plain OpenCV rectangle. `DrawingUtils.draw_main_bbox()` uses asset images from:

- `nedo_vision_worker_core/drawing_assets/blue/*`
- `nedo_vision_worker_core/drawing_assets/red/*`

It loads corner and line PNG assets, scales them based on frame height, and blends them onto the image with alpha. For small boxes, it falls back to simple corner lines.

### 6.4 Inner Attribute Boxes

`DrawingUtils.draw_inner_box()` is used for attribute-level PPE boxes. It overlays a texture frame inside the attribute bounding box and draws corner lines using a style derived from the label.

### 6.5 Cropped Image Box Alignment

`DrawingUtils.crop_with_bounding_box()` crops the person region with padding, rescales it, and transforms both the main bbox and attribute bboxes into the new cropped image coordinate space before drawing. This is why cropped images still show properly aligned boxes.

Crop transform logic:

1. Read original person bbox `[x1, y1, x2, y2]`
2. Expand crop window with a configurable buffer, clamped to image borders
3. Crop the region from the source frame
4. Resize the crop to a target height
5. Transform every bbox into crop-local coordinates:

```text
nx1 = (x1 - crop_x1) * scale
ny1 = (y1 - crop_y1) * scale
nx2 = (x2 - crop_x1) * scale
ny2 = (y2 - crop_y1) * scale
```

6. Return the resized cropped image and the transformed object
7. Draw again on the cropped image using the transformed bbox values

This is why the cropped violation image contains correctly placed person and attribute boxes rather than reusing the original full-frame coordinates.

### 6.6 End-to-End Flow Until Violation Image Is Captured

This is the full flow for a saved violation image:

```text
Video source
  -> VideoStreamManager.get_frame()
  -> PipelineProcessor._process_frame()
  -> detector.detect_objects(frame)
  -> detections: {label, confidence, bbox}
  -> detection processor matches detections to people
     - PPE: overlap coverage between attribute bbox and person bbox
     - Area: center point of person bbox inside polygon
  -> TrackerManager assigns UUID and updates attribute presence counters
  -> tracked object gains attr["count"]
  -> detection worker checks config flags
  -> repository checks trigger threshold count == 5
  -> repository renders full annotated image with FrameDrawer.draw_frame()
  -> repository crops person with DrawingUtils.crop_with_bounding_box()
  -> repository renders cropped annotated image with FrameDrawer.draw_frame()
  -> repository saves JPG files with cv2.imwrite()
  -> repository inserts DB rows containing paths and bbox columns
```

For restricted-area flow, `frame_drawer.draw_polygons(frame)` is called before full-frame save so the saved image also shows the configured area boundary.

## 7. Drawing Dependencies and Reuse

`DrawingUtils` is the more portable class. It mostly depends on standard Python modules plus `cv2` and `numpy`, and it expects drawing asset files.

`FrameDrawer` depends on `DrawingUtils`, asset files, icon files, and a specific tracked-object schema:

- `bbox`
- `track_id`
- `confidence`
- `attributes` with `label` and optional `bbox`

This means `FrameDrawer` is reusable in another project, but usually requires a small adapter layer and the matching assets directory structure.

## 8. Files Most Relevant To This Investigation

- `nedo_vision_worker_core/core_service.py`
- `nedo_vision_worker_core/database/DatabaseManager.py`
- `nedo_vision_worker_core/streams/StreamSyncThread.py`
- `nedo_vision_worker_core/streams/VideoStreamManager.py`
- `nedo_vision_worker_core/streams/VideoStream.py`
- `nedo_vision_worker_core/streams/SharedVideoDeviceManager.py`
- `nedo_vision_worker_core/services/VideoSharingDaemon.py`
- `nedo_vision_worker_core/pipeline/PipelineSyncThread.py`
- `nedo_vision_worker_core/pipeline/PipelineManager.py`
- `nedo_vision_worker_core/pipeline/PipelineProcessor.py`
- `nedo_vision_worker_core/pipeline/PipelineConfigManager.py`
- `nedo_vision_worker_core/tracker/TrackerManager.py`
- `nedo_vision_worker_core/util/PersonAttributeMatcher.py`
- `nedo_vision_worker_core/util/PersonRestrictedAreaMatcher.py`
- `nedo_vision_worker_core/ai/FrameDrawer.py`
- `nedo_vision_worker_core/util/DrawingUtils.py`
- `nedo_vision_worker_core/repositories/PPEDetectionRepository.py`
- `nedo_vision_worker_core/repositories/RestrictedAreaRepository.py`
- `nedo_vision_worker_core/repositories/WorkerSourcePipelineDetectionRepository.py`
- `nedo_vision_worker_core/repositories/WorkerSourcePipelineDebugRepository.py`
- `nedo_vision_worker_core/models/ppe_detection.py`
- `nedo_vision_worker_core/models/ppe_detection_label.py`
- `nedo_vision_worker_core/models/restricted_area_violation.py`
- `nedo_vision_worker_core/models/worker_source_pipeline_detection.py`
- `nedo_vision_worker_core/models/worker_source_pipeline_debug.py`

## 9. Key Conclusions

- The project uses local filesystem storage and local SQLite databases, not object storage.
- Violation capture is stability-based and usually happens after about five consecutive detection cycles.
- The saved images are annotated before being persisted.
- The drawing layer is centered on `FrameDrawer` and `DrawingUtils`.
- `DrawingUtils` is closer to portable; `FrameDrawer` is reusable but expects the project’s tracked-object schema and assets.
