class ModelPreparationError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")

