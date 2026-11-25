"""Isolated Python reference process. Never imports or times Pyre via Python."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys
import time

from .schema import package_version


class Unsupported(Exception):
    pass


def adapter(request, x):
    device, engine = request["device"], request["engine"]
    model = request["model"]
    threads = request["cpu_threads"] or 1
    prepared = request.get("native_input", "prepared") == "prepared"
    if engine == "nvforest":
        import nvforest
        import treelite
        tl_model = treelite.Model.deserialize(request["model_path"])
        forest = nvforest.load_from_treelite_model(tl_model, device=device, layout="depth_first",
                    align_bytes=0, precision="single" if model["model_dtype"] == "float32" else "double")
        if device == "gpu":
            import cupy as cp
            x = cp.asarray(x)
            cp.cuda.runtime.deviceSynchronize()
            # nvForest 26.6's predict already synchronizes its RAFT handle.
            # Avoid adding a second, device-wide barrier to every measured call.
            sync = lambda: None
            host = cp.asnumpy
        else:
            sync, host = lambda: None, lambda y: y
        return lambda: forest.predict(x, chunk_size=request["nvforest_chunk"]), sync, host, nvforest.__version__
    if engine != "native":
        raise ValueError(f"Unknown reference engine {engine}")
    if device == "gpu":
        if model["framework"] != "xgboost" or not prepared:
            raise Unsupported("Resident native GPU prediction requires prepared XGBoost inputs")
        import cupy as cp
        import xgboost as xgb
        if not xgb.build_info().get("USE_CUDA", False):
            raise RuntimeError("Install CUDA-enabled xgboost, not xgboost-cpu")
        booster = xgb.Booster()
        booster.load_model(request["native_path"])
        booster.set_param({"device": "cuda:0", "nthread": threads})
        x = cp.asarray(x)
        cp.cuda.runtime.deviceSynchronize()
        def predict():
            result = booster.inplace_predict(x)
            if not isinstance(result, cp.ndarray):
                raise RuntimeError("XGBoost did not return device output")
            return result
        return predict, cp.cuda.runtime.deviceSynchronize, cp.asnumpy, package_version("xgboost")
    framework = model["framework"]
    path = request["native_path"]
    if framework == "scikit-learn":
        import joblib
        estimator = joblib.load(path)  # Only locally generated corpus artifacts.
        estimator.set_params(n_jobs=threads)
        if prepared:
            x = x.astype("float32", copy=False)
        predict = lambda: estimator.predict(x)
        package = "scikit-learn"
    elif framework == "xgboost":
        import xgboost as xgb
        estimator = xgb.XGBRegressor(n_jobs=threads)
        estimator.load_model(path)
        estimator.set_params(n_jobs=threads, device="cpu")
        if prepared:
            booster = estimator.get_booster()
            predict = lambda: booster.inplace_predict(x)
        else:
            predict = lambda: estimator.predict(x)
        package = "xgboost-cpu"
    elif framework == "catboost":
        import catboost
        estimator = catboost.CatBoostRegressor()
        estimator.load_model(path)
        if prepared:
            x = catboost.Pool(x, thread_count=threads)
        predict = lambda: estimator.predict(x, thread_count=threads)
        package = "catboost"
    elif framework == "lightgbm":
        import lightgbm as lgb
        estimator = lgb.Booster(model_file=path)
        predict = lambda: estimator.predict(x, num_threads=threads)
        package = "lightgbm"
    else:
        raise Unsupported(f"No native adapter for {framework}")
    return predict, lambda: None, lambda y: y, package_version(package)


def main():
    import numpy as np
    from threadpoolctl import threadpool_limits
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        with threadpool_limits(limits=request["cpu_threads"] or 1):
            x = np.fromfile(request["input_path"], dtype=request["model"]["model_dtype"]).reshape(
                request["batch_rows"], request["model"]["features"])
            expected = np.fromfile(request["oracle_path"], dtype="float64")
            predict, sync, host, version = adapter(request, x)
            y = predict()
            sync()
            actual = np.asarray(host(y)).reshape(-1)
            if actual.shape != expected.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
                raise ValueError("Oracle shape/nonfinite failure")
            absolute = np.abs(actual.astype("float64") - expected)
            scaled = absolute / (1 + np.abs(expected))
            if np.any(scaled > request["tolerance"]):
                raise ValueError(f"Oracle mismatch: max scaled error {scaled.max()}")
            meta = dict(engine_version=version, output_dtype=str(actual.dtype),
                        max_abs_error=float(absolute.max()), max_scaled_error=float(scaled.max()),
                        api_layer="python", route=None, effective_row_group=request["nvforest_chunk"])
            settings = (dict(layout="depth_first", align_bytes=0, precision=request["model"]["model_dtype"],
                             chunk_size=request["nvforest_chunk"])
                        if request["engine"] == "nvforest" else dict(cpu_thread_limit=request["cpu_threads"], native_input=request.get("native_input", "prepared"),
                                  input_preparation="outside timed calls" if request.get("native_input", "prepared") == "prepared" else "native ingestion inside timed calls"))
            settings["device"] = request["device"]
            if request["device"] == "gpu":
                settings["input_preparation"] = "device upload outside timed calls"
                settings["completion"] = "nvforest handle sync" if request["engine"] == "nvforest" else "device synchronize per call"
            meta["engine_settings_json"] = json.dumps(settings, sort_keys=True)
            del y, actual

            def interval(loops):
                start = time.perf_counter_ns()
                for _ in range(loops):
                    result = predict()
                    sync()
                    del result
                return time.perf_counter_ns() - start

            interval(3)
            loops = 1
            while interval(loops) < request["min_seconds"] * 1e9:
                loops *= 2
                if loops > 1 << 28:
                    raise ValueError("Calibration exceeded loop budget")
            for sample in range(request["samples"]):
                elapsed = interval(loops)
                print(json.dumps(meta | dict(sample=sample, loops=loops, elapsed_ns=elapsed)), flush=True)
    except Unsupported as error:
        print(json.dumps({"status": "unsupported", "reason": str(error)}), flush=True)


if __name__ == "__main__":
    main()
