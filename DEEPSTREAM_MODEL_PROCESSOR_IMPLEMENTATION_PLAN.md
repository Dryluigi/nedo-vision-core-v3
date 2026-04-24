# DeepStream Model Processor Implementation Plan (RFD ETR First)

## 1. Goals
1. Accept `.pth` model from app data (`/app/raw_models`).
2. Auto-prepare DeepStream-ready artifacts on first `request_model_access`.
3. Reuse prepared artifacts on subsequent requests.
4. No database schema changes.
5. Keep architecture open for future YOLO processor.

## 2. Scope (Phase 1)
1. Implement only `rfdetr` processor.
2. Integrate processor call into `TritonModelManager.request_model_access`.
3. Generate model-scoped Triton/DeepStream config files.
4. Keep current runtime flow for owner load/unload unchanged.

## 3. High-Level Flow
1. `request_model_access(client_id, model_id, ai_model_id)` is called.
2. Manager resolves AI model (`type`, `file`) from existing `ai_model` table.
3. Manager picks processor by `type` (`rfdetr`).
4. Processor checks readiness of model artifacts.
5. If not ready, processor runs:
   1. `.pth -> .onnx` using `convert_rfdetr_auto.py`.
   2. `.onnx -> .plan` using `trtexec`.
   3. Generate `rfdetr_trt_*`, `rfdetr_ensemble_*`, and infer config files.
6. Manager starts `TritonModelOwner` with generated infer config path.
7. If ready already, skip build and load immediately.

## 4. Artifact Layout
1. Source checkpoint:
   1. `/app/raw_models/{ai_model.file}.pth`
2. ONNX:
   1. `/app/raw_models/{ai_model.file}.onnx`
3. TensorRT model:
   1. `/app/models/rfdetr_trt_{ai_model_id}/1/model.plan`
   2. `/app/models/rfdetr_trt_{ai_model_id}/config.pbtxt`
4. Ensemble model:
   1. `/app/models/rfdetr_ensemble_{ai_model_id}/1/`
   2. `/app/models/rfdetr_ensemble_{ai_model_id}/config.pbtxt`
5. DeepStream infer config:
   1. `/app/config/deepstream-inferserver-rfdetr-{ai_model_id}.txt`
6. Labels:
   1. `/app/models/rfdetr_{ai_model_id}/labels.txt` (canonical runtime labels)
   2. `/app/config/rfdetr-labels-{ai_model_id}.txt` (infer config label path)

## 5. Labels Mapping Plan (No DB Changes)
1. Use converter output `labels.txt` as source of truth.
2. Copy labels into model-scoped runtime location.
3. Generate model-scoped DeepStream label file from same content.
4. Infer config points to model-scoped label file.
5. Keep index order identical end-to-end (line index == class index expected by parser).

## 6. New Components
1. `DeepstreamModelProcessorInterface`
   1. `is_ready(ai_model) -> bool`
   2. `prepare(ai_model) -> PreparedModelPaths`
2. `RfdetrDeepstreamModelProcessor`
   1. Implements conversion/build/config generation.
3. `AIModelRepository` (read-only, existing table)
   1. Fetch by `id`.
4. `PreparedModelPaths` DTO
   1. Holds infer config path, model names, labels path, artifact paths.

## 7. Manager Adjustments
1. Add processor registry in manager:
   1. `{"rfdetr": RfdetrDeepstreamModelProcessor}`
2. Add per-`ai_model_id` preparation lock.
3. Resolve infer config dynamically from prepared output.
4. Keep owner lifecycle logic as-is after config resolution.

## 8. Readiness Rules
1. Ready only if all required files exist and non-empty:
   1. `.pth`, `.onnx`, `.plan`, Triton configs, ensemble version dir, infer config, labels.
2. Maintain filesystem manifest:
   1. `/app/models/rfdetr_{ai_model_id}/build_manifest.json`
3. Manifest includes:
   1. source `.pth` path
   2. source size/mtime
   3. chosen variant
   4. image size
   5. command metadata
4. Rebuild when source changed or required artifact missing.

## 9. Error Handling
1. Fail fast with stage-specific errors:
   1. `onnx_export_failed`
   2. `trt_build_failed`
   3. `config_generation_failed`
   4. `labels_mapping_failed`
2. Log command, exit code, stderr per stage.
3. Do not start owner when preparation fails.

## 9A. Failure Propagation Requirement
1. If model preparation fails for a requested model, `request_model_access` must fail immediately.
2. `TritonModelManager.request_model_access(...)` must re-raise preparation errors (do not swallow).
3. Pipeline startup (`_play_background`) must catch this error and publish stop event immediately.
4. Listener-visible payload must include:
   1. `status: stop`
   2. clear failure message (example: `Pipeline error: model preparation failed (trt_build_failed)`).
5. Startup thread must return immediately after publishing the error (no transition to `PLAYING`).

## 10. Extensibility for YOLO
1. Add `YoloDeepstreamModelProcessor` implementing same interface.
2. Add templates for YOLO trt/ensemble/infer config.
3. Register in processor registry by `ai_model.type`.
4. No manager flow rewrite needed.

## 11. Testing Plan
1. Unit tests:
   1. readiness detection
   2. label file generation
   3. config template rendering
2. Integration tests:
   1. first access triggers prepare
   2. second access skips prepare
   3. missing artifact forces rebuild
3. Manual smoke:
   1. run file pipeline with new prepared `rfdetr` model
   2. verify Triton loads generated ensemble and pipeline reaches `run`

## 12. Rollout Sequence
1. Add processor interface + rfdetr processor.
2. Add read-only AI model repository usage in manager.
3. Wire `request_model_access` pre-check and prepare.
4. Add manifest + readiness checks.
5. Validate with single test model end-to-end.
6. Enable for all rfdetr models.
