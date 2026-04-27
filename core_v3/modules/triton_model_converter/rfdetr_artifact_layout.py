from pathlib import Path


DEFAULT_MODEL_ARTIFACT_ROOT = Path("/app/model_artifacts")
DEFAULT_SHARED_MODEL_ROOT = Path("/app/models")


def get_rfdetr_artifact_paths(
    ai_model_id: str,
    artifact_root: str | Path = DEFAULT_MODEL_ARTIFACT_ROOT,
    shared_model_root: str | Path = DEFAULT_SHARED_MODEL_ROOT,
) -> dict[str, Path | str]:
    artifact_root = Path(artifact_root)
    shared_model_root = Path(shared_model_root)

    model_key = ai_model_id
    artifact_dir = artifact_root / "rfdetr" / model_key
    triton_repo_root = artifact_dir / "triton"
    config_root = artifact_dir / "config"

    trt_model_name = f"rfdetr_trt_{model_key}"
    ensemble_model_name = f"rfdetr_ensemble_{model_key}"

    trt_model_dir = triton_repo_root / trt_model_name
    trt_model_version_dir = trt_model_dir / "1"

    ensemble_model_dir = triton_repo_root / ensemble_model_name
    ensemble_model_version_dir = ensemble_model_dir / "1"

    postprocess_model_dir = triton_repo_root / "postprocess_rfdetr"
    shared_postprocess_model_dir = shared_model_root / "postprocess_rfdetr"

    return {
        "model_key": model_key,
        "artifact_dir": artifact_dir,
        "triton_repo_root": triton_repo_root,
        "config_root": config_root,
        "trt_model_name": trt_model_name,
        "trt_model_dir": trt_model_dir,
        "trt_model_version_dir": trt_model_version_dir,
        "ensemble_model_name": ensemble_model_name,
        "ensemble_model_dir": ensemble_model_dir,
        "ensemble_model_version_dir": ensemble_model_version_dir,
        "postprocess_model_dir": postprocess_model_dir,
        "shared_postprocess_model_dir": shared_postprocess_model_dir,
        "infer_config_path": config_root / f"deepstream-inferserver-rfdetr-{model_key}.txt",
        "labels_path": config_root / f"rfdetr-labels-{model_key}.txt",
        "manifest_path": config_root / f"rfdetr-manifest-{model_key}.json",
        "inprogress_path": config_root / f"rfdetr-build-{model_key}.inprogress",
    }


def get_rfdetr_infer_config_path(
    ai_model_id: str,
    artifact_root: str | Path = DEFAULT_MODEL_ARTIFACT_ROOT,
    shared_model_root: str | Path = DEFAULT_SHARED_MODEL_ROOT,
) -> str:
    paths = get_rfdetr_artifact_paths(
        ai_model_id=ai_model_id,
        artifact_root=artifact_root,
        shared_model_root=shared_model_root,
    )
    return str(paths["infer_config_path"])
