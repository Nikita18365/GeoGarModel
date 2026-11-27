"""Explicit compute policy. GPU selection must never silently become CPU training."""
import json

GPU_MODELS = {"MLP (pytorch)", "XGBoost", "LightGBM", "CatBoost", "ResNet", "Transformer"}
RESERVED = {"device", "device_type", "task_type", "devices", "gpu_device_id", "gpu_platform_id"}


def device_parameters(name, device="cpu", gpu_id=0):
    if device not in {"cpu", "gpu"} or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ValueError("Выберите CPU или GPU и неотрицательный номер GPU.")
    if device == "gpu" and name not in GPU_MODELS:
        raise ValueError(f"{name} поддерживает только CPU. Выберите GPU-модель или CPU.")
    if name == "XGBoost":
        return {"device": f"cuda:{gpu_id}" if device == "gpu" else "cpu", "tree_method": "hist"}
    if name == "CatBoost":
        return {"task_type": "GPU", "devices": str(gpu_id)} if device == "gpu" else {"task_type": "CPU"}
    if name == "LightGBM":
        return {"device_type": "gpu", "gpu_device_id": gpu_id, "force_col_wise": False} if device == "gpu" else {"device_type": "cpu"}
    if name in {"MLP (pytorch)", "ResNet", "Transformer"}:
        return {"device": f"cuda:{gpu_id}" if device == "gpu" else "cpu"}
    return {}


def verify_device(model, name, device):
    if device == "gpu" and name == "XGBoost":
        estimator = model.regressor_.steps[-1][1]
        config = json.loads(estimator.get_booster().save_config())
        actual = config["learner"]["generic_param"]["device"]
        if not actual.startswith("cuda"):
            raise RuntimeError("XGBoost перешёл на CPU: проверьте CUDA, драйвер и сборку XGBoost.")


class ComputationCancelled(Exception):
    """Cooperative cancellation, never forcibly terminates native training."""


def release_inversion_memory(gpu_id):
    """Run on the worker thread: cuFFT plans are per device and per thread."""
    import cupy as cp
    with cp.cuda.Device(gpu_id):
        try:
            cp.cuda.get_current_stream().synchronize()
        finally:
            try:
                cp.fft.config.get_plan_cache().clear()
            finally:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()


def check_cancel(event):
    if event is not None and event.is_set():
        raise ComputationCancelled("Расчёт отменён пользователем.")
