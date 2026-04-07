import time
from collections import defaultdict
from typing import List, Dict


class CaptureDecisionEngine:
    """
    Decides when a tracked person should trigger a capture event
    based on attribute persistence across frames.

    Each pipeline should create its own instance.
    """

    def __init__(
        self,
        capture_threshold: int = 5,
        track_timeout_seconds: float = 5.0,
    ):
        self.capture_threshold = capture_threshold
        self.track_timeout_seconds = track_timeout_seconds

        # person_id -> {attribute_label -> frame_count}
        self.attribute_counters: Dict[str, Dict] = defaultdict(lambda: defaultdict(int))
        self.track_counts: Dict[str, int] = defaultdict(int)

        # person_id -> last seen timestamp
        self.last_seen: Dict[str, float] = {}

        # person_id -> attributes already captured
        self.captured_flags: Dict[str, set] = defaultdict(set)

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def register(self, person_id: str, attributes: List):
        """
        Register detected attributes for a person in the current frame.

        Args:
            person_id: tracker id (or uuid)
            attributes: list of attribute labels detected this frame
        """
        now = time.time()

        current_attrs = set(attributes)
        counters = self.attribute_counters[person_id]

        # reset attributes not seen in this frame
        for label in list(counters.keys()):
            if label not in current_attrs:
                counters[label] = 0

        # increment current attributes
        for label in current_attrs:
            counters[label] += 1

        self.track_counts[person_id] += 1
        self.last_seen[person_id] = now

        self._cleanup()

    def should_capture(self, person_id: str) -> bool:
        """
        Returns True if this person should trigger a capture event.
        """
        counters = self.attribute_counters.get(person_id)
        if not counters:
            return False

        for label, count in counters.items():

            if count == self.capture_threshold:

                # avoid duplicate capture
                if label not in self.captured_flags[person_id]:
                    self.captured_flags[person_id].add(label)
                    return True

        return False

    def get_attribute_counts(self, person_id: str) -> Dict:
        return dict(self.attribute_counters.get(person_id, {}))

    def get_detection_count(self, person_id: str) -> int:
        return self.track_counts.get(person_id, 0)

    def get_triggered_labels(self, person_id: str):
        counters = self.attribute_counters.get(person_id)
        if not counters:
            return []

        triggered_labels = []
        for label, count in counters.items():
            if count == self.capture_threshold and label not in self.captured_flags[person_id]:
                self.captured_flags[person_id].add(label)
                triggered_labels.append(label)

        return triggered_labels

    # -------------------------------------------------
    # INTERNAL
    # -------------------------------------------------

    def _cleanup(self):
        """
        Remove stale persons that disappeared from the scene.
        """
        now = time.time()

        expired = [
            pid
            for pid, last_seen in self.last_seen.items()
            if now - last_seen > self.track_timeout_seconds
        ]

        for pid in expired:
            self.attribute_counters.pop(pid, None)
            self.track_counts.pop(pid, None)
            self.last_seen.pop(pid, None)
            self.captured_flags.pop(pid, None)
