import threading
from typing import Dict, Set, Optional

from .triton_model_owner import TritonModelOwner


# ── Config Registry ───────────────────────────────────────────────────────────

MODEL_CONFIG_MAP: Dict[str, str] = {
    "yolo":   "/app/config/deepstream-inferserver-yolo.txt",
    "rfdetr": "/app/config/deepstream-inferserver-rfdetr.txt",
}

# ── Manager ───────────────────────────────────────────────────────────────────

class TritonModelManager:
    """
    Manages multiple TritonModelOwner instances.

    - Lazily creates an owner on first request_model_access()
    - Tracks which client_id is using which model_ids
    - Starts idle countdown when last client releases a model
    - Unloads model after idle_timeout_seconds of no clients
    """

    def __init__(self, gpu_id: int = 0, idle_timeout_seconds: float = 60.0):
        self._gpu_id               = gpu_id
        self._idle_timeout         = idle_timeout_seconds
        self._lock                 = threading.Lock()

        # model_id → TritonModelOwner
        self._owners: Dict[str, TritonModelOwner] = {}

        # model_id → set of active client_ids
        self._clients: Dict[str, Set[str]] = {}

        # client_id → set of model_ids it is using
        self._client_models: Dict[str, Set[str]] = {}

        # model_id → idle timer thread
        self._idle_timers: Dict[str, threading.Timer] = {}

    def request_model_access(self, client_id: str, model_id: str):
        """
        Register a client as using a model.
        Lazily starts the TritonModelOwner if not already running.
        Cancels any pending idle shutdown timer for that model.
        """
        with self._lock:
            # Cancel idle timer if model was counting down
            self._cancel_idle_timer(model_id)

            if model_id not in self._owners:
                self._start_owner(model_id)

            self._clients[model_id].add(client_id)

            if client_id not in self._client_models:
                self._client_models[client_id] = set()
            self._client_models[client_id].add(model_id)

            print(
                f"[TritonModelManager] Client '{client_id}' registered for model '{model_id}' "
                f"(active clients: {len(self._clients[model_id])})"
            )

    def release_model_access(self, client_id: str):
        """
        Deregister a client from all models it was using.
        Starts idle countdown for any model that now has no clients.
        """
        with self._lock:
            model_ids = self._client_models.pop(client_id, set())

            if not model_ids:
                print(f"[TritonModelManager] Client '{client_id}' had no registered models")
                return

            for model_id in model_ids:
                if model_id not in self._clients:
                    continue

                self._clients[model_id].discard(client_id)
                remaining = len(self._clients[model_id])

                print(
                    f"[TritonModelManager] Client '{client_id}' released model '{model_id}' "
                    f"(remaining clients: {remaining})"
                )

                if remaining == 0:
                    self._schedule_idle_shutdown(model_id)

    def wait_model_till_ready(self, model_id: str, timeout: Optional[float] = None):
        """
        Block until the model is confirmed ready (first buffer through fakesink).
        Call this after request_model_access(), before starting inference pipeline.
        """
        with self._lock:
            owner = self._owners.get(model_id)

        if not owner:
            raise RuntimeError(
                f"[TritonModelManager] Model '{model_id}' not registered. "
                f"Call request_model_access() first."
            )

        owner.wait_till_ready(timeout=timeout)

    def is_model_ready(self, model_id: str) -> bool:
        with self._lock:
            owner = self._owners.get(model_id)
        return owner.is_ready() if owner else False

    def shutdown(self):
        """Unload all models — call on application exit."""
        with self._lock:
            # Cancel all pending timers first
            for model_id in list(self._idle_timers.keys()):
                self._cancel_idle_timer(model_id)

            for model_id in list(self._owners.keys()):
                self._stop_owner(model_id)

        print("[TritonModelManager] All models unloaded")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _start_owner(self, model_id: str):
        """Must be called within self._lock."""
        if model_id not in MODEL_CONFIG_MAP:
            raise ValueError(
                f"[TritonModelManager] Unknown model_id '{model_id}'. "
                f"Available: {list(MODEL_CONFIG_MAP.keys())}"
            )

        print(f"[TritonModelManager] Starting owner for model '{model_id}'")
        owner = TritonModelOwner(
            model_id=model_id,
            infer_config=MODEL_CONFIG_MAP[model_id],
            gpu_id=self._gpu_id
        )
        owner.load()
        self._owners[model_id] = owner
        self._clients[model_id] = set()

    def _stop_owner(self, model_id: str):
        """Must be called within self._lock."""
        owner = self._owners.pop(model_id, None)
        self._clients.pop(model_id, None)

        if owner:
            threading.Thread(
                target=owner.unload,
                daemon=True,
                name=f"triton-unload-{model_id}"
            ).start()

    def _schedule_idle_shutdown(self, model_id: str):
        """Must be called within self._lock. Starts idle countdown for a model."""
        print(
            f"[TritonModelManager] No clients for model '{model_id}'. "
            f"Scheduling shutdown in {self._idle_timeout}s"
        )

        timer = threading.Timer(
            interval=self._idle_timeout,
            function=self._idle_shutdown,
            args=(model_id,)
        )
        timer.daemon = True
        timer.start()
        self._idle_timers[model_id] = timer

    def _cancel_idle_timer(self, model_id: str):
        """Must be called within self._lock."""
        timer = self._idle_timers.pop(model_id, None)
        if timer:
            timer.cancel()
            print(f"[TritonModelManager] Cancelled idle shutdown for model '{model_id}'")

    def _idle_shutdown(self, model_id: str):
        """Called by Timer thread after idle_timeout_seconds."""
        with self._lock:
            # Double-check no new client registered during the timer window
            active_clients = self._clients.get(model_id, set())
            if active_clients:
                print(f"[TritonModelManager] Idle timer fired but model '{model_id}' has active clients — aborting shutdown")
                return

            self._idle_timers.pop(model_id, None)
            print(f"[TritonModelManager] Idle timeout reached — unloading model '{model_id}'")
            self._stop_owner(model_id)