import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from distress_detector import DistressDetector, CRISIS_MESSAGE, CHECK_IN_MESSAGE

config = os.getenv("DETECTOR_CONFIG", "./results/calibration/detector_config.json")
detector = DistressDetector(
    os.getenv("MODEL_PATH", "./models/resultsroberta/checkpoint-258"),
    tokenizer_path=os.getenv("TOKENIZER_PATH", "./models/resultsroberta"),
    config_path=config if os.path.exists(config) else None,
)

app = FastAPI(title="Distress detection component")


class Request(BaseModel):
    message: str
    history: Optional[List[str]] = None


@app.post("/assess")
def assess(req: Request):
    result = detector.assess(req.message, history=req.history)
    if result["action"] == "escalate":
        result["suggested_reply"] = CRISIS_MESSAGE
    elif result["action"] == "check_in":
        result["suggested_reply"] = CHECK_IN_MESSAGE
    else:
        result["suggested_reply"] = None
    return result


@app.get("/health")
def health():
    return {"status": "ok", "temperature": detector.temperature, "high_risk_threshold": detector.high_risk_threshold}
