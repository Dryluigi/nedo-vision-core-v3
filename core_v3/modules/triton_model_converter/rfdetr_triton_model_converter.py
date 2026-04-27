import json
import subprocess
from pathlib import Path
import logging

from ...models.ai_model import AIModelEntity
from .triton_model_converter_interface import TritonModelConverterInterface
from .model_preparation_error import ModelPreparationError


class RfdetrTritonModelConverter(TritonModelConverterInterface):
    def __init__(
        self,
        raw_model_root: str = "/app/raw_models",
        converter_script_root: str = "/app/core_v3/scripts",
        triton_model_root: str = "/app/models",
        config_root: str = "/app/config",
        image_size: int = 640,
    ):
        self._raw_model_root = Path(raw_model_root)
        self._converter_script_root = Path(converter_script_root)
        self._triton_model_root = Path(triton_model_root)
        self._config_root = Path(config_root)
        self._image_size = image_size

    def is_ready(self, ai_model: AIModelEntity) -> bool:
        paths = self._build_paths(ai_model)
        if paths["inprogress_path"].exists():
            return False

        required_paths = [
            paths["weights_path"],
            paths["onnx_path"],
            paths["plan_path"],
            paths["trt_config_path"],
            paths["ensemble_config_path"],
            paths["infer_config_path"],
            paths["labels_path"],
            paths["manifest_path"],
        ]
        if not all(p.exists() and p.stat().st_size > 0 for p in required_paths):
            return False

        try:
            manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        try:
            weights_stat = paths["weights_path"].stat()
        except OSError:
            return False

        expected_source = str(paths["weights_path"])
        if manifest.get("source_weights_path") != expected_source:
            return False
        if int(manifest.get("source_size", -1)) != int(weights_stat.st_size):
            return False
        if int(manifest.get("image_size", -1)) != int(self._image_size):
            return False
        return int(manifest.get("source_mtime_ns", -1)) == int(weights_stat.st_mtime_ns)

    def prepare(self, ai_model: AIModelEntity):
        paths = self._build_paths(ai_model)
        weights_path = paths["weights_path"]
        if not weights_path.exists():
            raise ModelPreparationError(
                stage="missing_weights",
                message=f"Checkpoint not found: {weights_path}",
            )

        logging.info(
            "[RfdetrTritonModelConverter] Preparing model ai_model_id=%s from %s",
            ai_model.id,
            weights_path,
        )

        paths["trt_model_dir"].mkdir(parents=True, exist_ok=True)
        paths["trt_model_version_dir"].mkdir(parents=True, exist_ok=True)
        paths["ensemble_model_dir"].mkdir(parents=True, exist_ok=True)
        paths["ensemble_model_version_dir"].mkdir(parents=True, exist_ok=True)
        self._config_root.mkdir(parents=True, exist_ok=True)

        self._cleanup_temp_artifacts(paths)
        paths["inprogress_path"].write_text("building", encoding="utf-8")

        try:
            logging.info(
                "[RfdetrTritonModelConverter] Running ONNX export script -> %s",
                paths["onnx_tmp_path"],
            )
            self._run(
                [
                    "python3",
                    "convert_rfdetr_auto.py",
                    "-w",
                    str(weights_path),
                    "-s",
                    str(self._image_size),
                    "--dynamic",
                    "--output",
                    str(paths["onnx_tmp_path"]),
                    "--labels-output",
                    str(paths["labels_tmp_path"]),
                ],
                cwd=self._converter_script_root,
                stage="onnx_export_failed",
            )

            logging.info(
                "[RfdetrTritonModelConverter] Building TensorRT engine -> %s",
                paths["plan_tmp_path"],
            )
            self._run(
                [
                    "trtexec",
                    f"--onnx={paths['onnx_tmp_path']}",
                    f"--saveEngine={paths['plan_tmp_path']}",
                    f"--shapes=input:1x3x{self._image_size}x{self._image_size}",
                    "--fp16",
                ],
                cwd=Path("/app"),
                stage="trt_build_failed",
            )

            logging.info(
                "[RfdetrTritonModelConverter] Writing Triton/DeepStream configs for ai_model_id=%s",
                ai_model.id,
            )
            self._atomic_write_text(
                paths["trt_config_tmp_path"],
                self._render_trt_config(
                    model_name=paths["trt_model_name"],
                    image_size=self._image_size,
                ),
            )
            self._atomic_write_text(
                paths["ensemble_config_tmp_path"],
                self._render_ensemble_config(
                    ensemble_name=paths["ensemble_model_name"],
                    trt_model_name=paths["trt_model_name"],
                    image_size=self._image_size,
                ),
            )
            self._atomic_write_text(
                paths["infer_config_tmp_path"],
                self._render_infer_config(
                    ensemble_model_name=paths["ensemble_model_name"],
                    labels_path=paths["labels_path"],
                ),
            )
            self._write_manifest(paths)
            self._promote_tmp_artifacts(paths)
        finally:
            self._cleanup_temp_artifacts(paths)
            if paths["inprogress_path"].exists():
                paths["inprogress_path"].unlink()

        logging.info(
            "[RfdetrTritonModelConverter] Preparation completed ai_model_id=%s infer_config=%s",
            ai_model.id,
            paths["infer_config_path"],
        )
        return {
            "infer_config_path": str(paths["infer_config_path"]),
            "ensemble_model_name": paths["ensemble_model_name"],
            "trt_model_name": paths["trt_model_name"],
            "labels_path": str(paths["labels_path"]),
        }

    def _build_paths(self, ai_model: AIModelEntity) -> dict:
        file_name = (ai_model.file or "").strip()
        if not file_name:
            raise ModelPreparationError(
                stage="invalid_ai_model_file",
                message=f"AI model file is empty for ai_model_id={ai_model.id}",
            )

        weights_path = self._raw_model_root / file_name
        if weights_path.suffix.lower() != ".pth":
            weights_path = weights_path.with_suffix(".pth")
        onnx_path = weights_path.with_suffix(".onnx")

        model_key = ai_model.id
        trt_model_name = f"rfdetr_trt_{model_key}"
        ensemble_model_name = f"rfdetr_ensemble_{model_key}"

        trt_model_dir = self._triton_model_root / trt_model_name
        trt_model_version_dir = trt_model_dir / "1"
        plan_path = trt_model_version_dir / "model.plan"
        plan_tmp_path = trt_model_version_dir / "model.plan.tmp"
        trt_config_path = trt_model_dir / "config.pbtxt"
        trt_config_tmp_path = trt_model_dir / "config.pbtxt.tmp"

        ensemble_model_dir = self._triton_model_root / ensemble_model_name
        ensemble_model_version_dir = ensemble_model_dir / "1"
        ensemble_config_path = ensemble_model_dir / "config.pbtxt"
        ensemble_config_tmp_path = ensemble_model_dir / "config.pbtxt.tmp"

        labels_path = self._config_root / f"rfdetr-labels-{model_key}.txt"
        labels_tmp_path = self._config_root / f"rfdetr-labels-{model_key}.txt.tmp"
        infer_config_path = self._config_root / f"deepstream-inferserver-rfdetr-{model_key}.txt"
        infer_config_tmp_path = self._config_root / f"deepstream-inferserver-rfdetr-{model_key}.txt.tmp"
        manifest_path = self._config_root / f"rfdetr-manifest-{model_key}.json"
        manifest_tmp_path = self._config_root / f"rfdetr-manifest-{model_key}.json.tmp"
        inprogress_path = self._config_root / f"rfdetr-build-{model_key}.inprogress"

        return {
            "weights_path": weights_path,
            "onnx_path": onnx_path,
            "onnx_tmp_path": onnx_path.with_suffix(".onnx.tmp"),
            "trt_model_name": trt_model_name,
            "trt_model_dir": trt_model_dir,
            "trt_model_version_dir": trt_model_version_dir,
            "plan_path": plan_path,
            "plan_tmp_path": plan_tmp_path,
            "trt_config_path": trt_config_path,
            "trt_config_tmp_path": trt_config_tmp_path,
            "ensemble_model_name": ensemble_model_name,
            "ensemble_model_dir": ensemble_model_dir,
            "ensemble_model_version_dir": ensemble_model_version_dir,
            "ensemble_config_path": ensemble_config_path,
            "ensemble_config_tmp_path": ensemble_config_tmp_path,
            "labels_path": labels_path,
            "labels_tmp_path": labels_tmp_path,
            "infer_config_path": infer_config_path,
            "infer_config_tmp_path": infer_config_tmp_path,
            "manifest_path": manifest_path,
            "manifest_tmp_path": manifest_tmp_path,
            "inprogress_path": inprogress_path,
        }

    def _atomic_write_text(self, path: Path, content: str):
        path.write_text(content, encoding="utf-8")
        if not path.exists() or path.stat().st_size == 0:
            raise ModelPreparationError(
                stage="write_config_failed",
                message=f"Failed to write file: {path}",
            )

    def _promote_tmp_artifacts(self, paths: dict):
        pairs = [
            ("onnx_tmp_path", "onnx_path"),
            ("labels_tmp_path", "labels_path"),
            ("plan_tmp_path", "plan_path"),
            ("trt_config_tmp_path", "trt_config_path"),
            ("ensemble_config_tmp_path", "ensemble_config_path"),
            ("infer_config_tmp_path", "infer_config_path"),
            ("manifest_tmp_path", "manifest_path"),
        ]
        for tmp_key, final_key in pairs:
            self._replace(paths[tmp_key], paths[final_key], stage="promote_artifacts_failed")

    def _write_manifest(self, paths: dict):
        weights_stat = paths["weights_path"].stat()
        manifest = {
            "source_weights_path": str(paths["weights_path"]),
            "source_size": int(weights_stat.st_size),
            "source_mtime_ns": int(weights_stat.st_mtime_ns),
            "image_size": int(self._image_size),
            "onnx_path": str(paths["onnx_path"]),
            "plan_path": str(paths["plan_path"]),
            "labels_path": str(paths["labels_path"]),
            "infer_config_path": str(paths["infer_config_path"]),
        }
        paths["manifest_tmp_path"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _replace(src: Path, dst: Path, stage: str):
        if not src.exists() or src.stat().st_size == 0:
            raise ModelPreparationError(
                stage=stage,
                message=f"Temporary artifact missing or empty: {src}",
            )
        src.replace(dst)

    @staticmethod
    def _cleanup_temp_artifacts(paths: dict):
        temp_paths = [
            paths["onnx_tmp_path"],
            paths["labels_tmp_path"],
            paths["plan_tmp_path"],
            paths["trt_config_tmp_path"],
            paths["ensemble_config_tmp_path"],
            paths["infer_config_tmp_path"],
            paths["manifest_tmp_path"],
        ]
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()

    def _run(self, cmd: list[str], cwd: Path, stage: str):
        logging.info(
            "[RfdetrTritonModelConverter] Execute (%s): %s",
            stage,
            " ".join(cmd),
        )
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        if proc.stdout.strip():
            logging.info(
                "[RfdetrTritonModelConverter] stdout (%s): %s",
                stage,
                proc.stdout.strip(),
            )
        if proc.stderr.strip():
            logging.info(
                "[RfdetrTritonModelConverter] stderr (%s): %s",
                stage,
                proc.stderr.strip(),
            )
        if proc.returncode != 0:
            raise ModelPreparationError(
                stage=stage,
                message=(
                    f"Command failed ({' '.join(cmd)}). "
                    f"stdout: {proc.stdout.strip()} | stderr: {proc.stderr.strip()}"
                ),
            )

    @staticmethod
    def _render_trt_config(model_name: str, image_size: int) -> str:
        return f"""name: "{model_name}"
backend: "tensorrt"
max_batch_size: 1
default_model_filename: "model.plan"

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, {image_size}, {image_size} ]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1, 6 ]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""

    @staticmethod
    def _render_ensemble_config(ensemble_name: str, trt_model_name: str, image_size: int) -> str:
        return f"""name: "{ensemble_name}"
platform: "ensemble"
max_batch_size: 1

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, {image_size}, {image_size} ]
  }}
]

output [
  {{
    name: "num_detections"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }},
  {{
    name: "detection_boxes"
    data_type: TYPE_FP32
    dims: [ 100, 4 ]
  }},
  {{
    name: "detection_scores"
    data_type: TYPE_FP32
    dims: [ 100 ]
  }},
  {{
    name: "detection_classes"
    data_type: TYPE_FP32
    dims: [ 100 ]
  }}
]

ensemble_scheduling {{
  step [
    {{
      model_name: "{trt_model_name}"
      model_version: -1
      input_map {{
        key: "input"
        value: "input"
      }}
      output_map {{
        key: "output"
        value: "rfdetr_raw_output"
      }}
    }},
    {{
      model_name: "postprocess_rfdetr"
      model_version: -1
      input_map {{
        key: "INPUT"
        value: "rfdetr_raw_output"
      }}
      output_map {{
        key: "num_detections"
        value: "num_detections"
      }},
      output_map {{
        key: "detection_boxes"
        value: "detection_boxes"
      }},
      output_map {{
        key: "detection_scores"
        value: "detection_scores"
      }},
      output_map {{
        key: "detection_classes"
        value: "detection_classes"
      }}
    }}
  ]
}}
"""

    @staticmethod
    def _render_infer_config(ensemble_model_name: str, labels_path: Path) -> str:
        return f"""infer_config {{
  unique_id: 2
  max_batch_size: 1

  backend {{
    triton {{
      model_name: "{ensemble_model_name}"
      version: -1
      model_repo {{
        root: "/app/models"
      }}
    }}
  }}

  preprocess {{
    network_format: IMAGE_FORMAT_RGB
    tensor_order: TENSOR_ORDER_LINEAR
    normalize {{
      scale_factor: 0.003921569
    }}
  }}

  custom_lib {{
    path: "/app/lib/custom_parser/libcustom_parser.so"
  }}

  postprocess {{
    labelfile_path: "{labels_path}"
    detection {{
      num_detected_classes: 80
      custom_parse_bbox_func: "NvDsInferParseCustomTritonYolo"
    }}
  }}
}}

input_control {{
  process_mode: PROCESS_MODE_FULL_FRAME
}}
"""
