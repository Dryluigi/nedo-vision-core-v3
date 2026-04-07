from ..models.ppe_detection import PPEDetectionEntity
from ..models.ppe_detection_label import PPEDetectionLabelEntity
from .BaseRepository import BaseRepository


class PPEDetectionRepository(BaseRepository):
    def __init__(self):
        super().__init__(db_name="default")

    def save_ppe_detection(
        self,
        worker_id: str,
        worker_source_id: str,
        person_id: str,
        image_path: str,
        image_tile_path: str,
        detection_count: int,
        person_bbox,
        attributes,
    ) -> str:
        with self._get_session() as session:
            detection = PPEDetectionEntity(
                worker_id=worker_id,
                worker_source_id=worker_source_id,
                person_id=person_id,
                image_path=image_path,
                image_tile_path=image_tile_path,
                detection_count=detection_count,
                b_box_x1=float(person_bbox[0]),
                b_box_y1=float(person_bbox[1]),
                b_box_x2=float(person_bbox[2]),
                b_box_y2=float(person_bbox[3]),
            )
            session.add(detection)
            session.flush()

            for attr in attributes:
                bbox = attr.get("bbox") or [0, 0, 0, 0]
                session.add(
                    PPEDetectionLabelEntity(
                        detection_id=detection.id,
                        code=attr["label"],
                        confidence_score=float(attr.get("confidence", 0.0)),
                        detection_count=int(attr.get("count", 0)),
                        b_box_x1=float(bbox[0]),
                        b_box_y1=float(bbox[1]),
                        b_box_x2=float(bbox[2]),
                        b_box_y2=float(bbox[3]),
                    )
                )

            return detection.id
