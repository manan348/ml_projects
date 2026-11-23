"""Optional MLflow experiment tracking (works with DagsHub).

Nothing here runs unless a script is started with --track, so the project
works without mlflow or dagshub installed.

Where runs are logged:
  * DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME set -> the DagsHub MLflow server
    of that repository (authentication: DAGSHUB_USER_TOKEN, or the browser
    login that dagshub opens on first use).
  * MLFLOW_TRACKING_URI set                       -> that server.
  * neither                                       -> a local mlflow.db file.
"""

import os
from contextlib import contextmanager

_mlflow = None  # set while a tracked run is active


@contextmanager
def tracking(enabled: bool, experiment: str, run_name: str):
    """Open an MLflow run if tracking is enabled, otherwise do nothing."""
    global _mlflow
    if not enabled:
        yield
        return
    try:
        import mlflow
    except ImportError:
        raise SystemExit(
            "--track needs extra packages: pip install -r requirements-dagshub.txt"
        )

    owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME")
    if owner and repo:
        import dagshub

        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        print(f"MLflow tracking on DagsHub: {owner}/{repo}")
    elif os.getenv("MLFLOW_TRACKING_URI"):
        print(f"MLflow tracking: {os.getenv('MLFLOW_TRACKING_URI')}")
    else:
        # Explicit local database: MLflow's implicit default misbehaves on
        # Windows paths with spaces and the plain ./mlruns file store is deprecated.
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        print("MLflow tracking: local mlflow.db in the current folder "
              "(view with: mlflow ui --backend-store-uri sqlite:///mlflow.db)")

    mlflow.set_experiment(experiment)
    _mlflow = mlflow
    try:
        with mlflow.start_run(run_name=run_name):
            yield
    finally:
        _mlflow = None


def log_params(params: dict):
    if _mlflow:
        _mlflow.log_params({k: str(v) for k, v in params.items()})


def log_metrics(metrics: dict):
    if _mlflow:
        _mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def log_artifacts(folder):
    if _mlflow:
        _mlflow.log_artifacts(str(folder))
