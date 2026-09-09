import os
import urllib3
from fastapi import FastAPI, HTTPException
from kfp.client import Client

# Відключаємо попередження SSL для взаємодії всередині кластера
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="Pipeline Status Utility Service")

# Читаємо налаштування з оточення
NAMESPACE = os.getenv("TARGET_NAMESPACE", "serhiizn-dev")

@app.get("/")
def root():
    return {
        "status": "active",
        "message": "Pipeline Status Knative Service is running",
    }

@app.get("/pipeline-status")
def get_latest_pipeline_status():
    try:
        client = Client(
            namespace=NAMESPACE,
        )

        runs = client.list_runs(page_size=5).runs
        if not runs:
            return {"namespace": NAMESPACE, "runs": [], "message": "No runs found"}

        result = []
        for run in runs:
            result.append({
                "run_id": run.id,
                "name": run.name,
                "status": run.state,
                "created_at": str(run.created_at),
            })

        return {"namespace": NAMESPACE, "latest_runs": result}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch pipeline status: {str(e)}"
        )
