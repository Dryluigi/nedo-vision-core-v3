from typing import List, Dict


class PersonAttributeAggregator:
    """
    Aggregates attribute detections (helmet, vest, etc.)
    to detected persons using bounding box overlap.

    Labels are integer class ids.
    """

    def __init__(
        self,
        person_class_id: int,
        attribute_class_ids: List[int],
        person_conf_threshold: float = 0.5,
        attribute_conf_threshold: float = 0.5,
        coverage_threshold: float = 0.3,
    ):
        self.person_class_id = person_class_id
        self.attribute_class_ids = set(attribute_class_ids)
        self.person_conf_threshold = person_conf_threshold
        self.attribute_conf_threshold = attribute_conf_threshold
        self.coverage_threshold = coverage_threshold

    # -------------------------------------------------
    # PUBLIC
    # -------------------------------------------------

    def aggregate(self, detections: List[Dict]) -> List[Dict]:
        """
        Input detection format:

        [
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": int,
                "confidence": float,
                "track_id": int (optional)
            }
        ]
        """

        persons = []
        attributes = []

        for det in detections:

            cid = det["class_id"]
            conf = det.get("confidence", 1.0)

            if cid == self.person_class_id and conf >= self.person_conf_threshold:
                persons.append(det)

            elif cid in self.attribute_class_ids and conf >= self.attribute_conf_threshold:
                attributes.append(det)

        results = []

        for person in persons:

            pbox = person["bbox"]
            assigned_attrs = []

            for attr in attributes:

                coverage = self._coverage(pbox, attr["bbox"])

                if coverage >= self.coverage_threshold:
                    assigned_attrs.append(attr)

            results.append({
                "person_id": person.get("track_id"),
                "bbox": pbox,
                "confidence": person.get("confidence", 1.0),
                "attributes": assigned_attrs,
            })

        return results

    # -------------------------------------------------
    # INTERNAL
    # -------------------------------------------------

    def _coverage(self, boxA, boxB):

        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)

        attr_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return inter_area / attr_area