import json
import time
import hashlib
import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("distress_detector")

LABELS = ["neutral", "moderate", "high-risk"]

ACTIONS = {
    "neutral": "continue",
    "moderate": "check_in",
    "high-risk": "escalate",
}

CRISIS_MESSAGE = (
    "It sounds like you are going through something really painful right now, and I'm concerned about your safety. "
    "I'm not able to give you the support you need in this situation, but people are available right now:\n"
    "- If you are in immediate danger, call 999 or go to A&E.\n"
    "- Samaritans: call 116 123 (free, 24/7).\n"
    "- Shout: text SHOUT to 85258 (free, 24/7).\n"
    "- NHS 111 (option 2 for mental health support in England).\n"
    "We can pause the exercises for now."
)

CHECK_IN_MESSAGE = (
    "Thank you for sharing that - it sounds difficult. Before we continue, how are you feeling right now? "
    "We can carry on with the exercise, take a break, or I can point you to other sources of support."
)


class DistressDetector:

    def __init__(self, model_path, tokenizer_path=None, config_path=None, device=None,
                 max_length=512, context_turns=3, batch_size=16, context_mode="replace"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.context_turns = context_turns
        self.batch_size = batch_size
        self.context_mode = context_mode

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.temperature = 1.0
        self.high_risk_threshold = None
        if config_path:
            with open(config_path) as f:
                cfg = json.load(f)
            self.temperature = cfg.get("temperature", 1.0)
            self.high_risk_threshold = cfg.get("high_risk_threshold")

    def logits(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if isinstance(t, str) and t.strip() else "" for t in texts[i:i + self.batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out.append(self.model(**enc).logits.cpu().numpy())
        return np.concatenate(out, axis=0)

    def probabilities(self, texts):
        z = self.logits(texts) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def decide(self, probs):
        if self.high_risk_threshold is not None and probs[2] >= self.high_risk_threshold:
            return "high-risk"
        if self.high_risk_threshold is not None:
            return LABELS[int(np.argmax(probs[:2]))]
        return LABELS[int(np.argmax(probs))]

    def assess(self, message, history=None):
        start = time.perf_counter()
        msg_hash = hashlib.sha256((message or "").encode()).hexdigest()[:12]

        if not message or not message.strip():
            return {"label": "neutral", "action": "continue", "probabilities": None,
                    "context_used": False, "latency_ms": 0.0}

        try:
            texts = [message]
            window = None
            if history and self.context_turns > 0 and self.context_mode != "none":
                window = " ".join(list(history[-self.context_turns:]) + [message])
                texts.append(window)

            probs_all = self.probabilities(texts)
            probs = probs_all[0].copy()
            context_used = False
            if window is not None and probs_all[1][2] > probs[2]:
                if self.context_mode == "max_high":
                    probs[2] = probs_all[1][2]
                else:
                    probs = probs_all[1].copy()
                context_used = True

            label = self.decide(probs)
            result = {
                "label": label,
                "action": ACTIONS[label],
                "probabilities": {LABELS[k]: float(probs[k]) for k in range(3)},
                "context_used": context_used,
            }
        except Exception as e:
            logger.exception("Detector failed on message %s: %s", msg_hash, e)
            result = {"label": "unknown", "action": "check_in", "probabilities": None, "context_used": False}

        result["latency_ms"] = (time.perf_counter() - start) * 1000
        logger.info("msg=%s label=%s action=%s latency_ms=%.1f",
                    msg_hash, result["label"], result["action"], result["latency_ms"])
        return result

    def is_high_risk(self, text):
        return self.assess(text)["label"] == "high-risk"
