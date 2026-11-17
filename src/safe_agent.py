from distress_detector import CRISIS_MESSAGE, CHECK_IN_MESSAGE


def new_session():
    return {"history": [], "in_crisis": False, "assessments": []}


def gated_reply(message, session, host_reply_fn, detector, gate_on=True):
    user_turns = [t for r, t in session["history"] if r == "user"]
    assessment = detector.assess(message, history=user_turns) if gate_on else None

    if not gate_on:
        reply = host_reply_fn(message, session["history"])
    elif assessment["action"] == "escalate":
        session["in_crisis"] = True
        reply = CRISIS_MESSAGE
    elif session["in_crisis"]:
        reply = "I'm still here. " + CRISIS_MESSAGE
    elif assessment["action"] == "check_in":
        reply = CHECK_IN_MESSAGE
    else:
        reply = host_reply_fn(message, session["history"])

    session["history"].append(("user", message))
    session["history"].append(("assistant", reply))
    session["assessments"].append(assessment)
    return reply, assessment


def stub_sat_host(message, history):
    return "Let's continue with today's Self-Attachment exercise. Picture your childhood photo..."


def make_openai_host(model="gpt-4o-mini", system_prompt="You are a supportive self-help coach."):
    from openai import OpenAI
    client = OpenAI()

    def host(message, history):
        msgs = [{"role": "system", "content": system_prompt}]
        msgs += [{"role": r, "content": t} for r, t in history]
        msgs.append({"role": "user", "content": message})
        out = client.chat.completions.create(model=model, messages=msgs, temperature=0)
        return out.choices[0].message.content
    return host


if __name__ == "__main__":
    from distress_detector import DistressDetector
    det = DistressDetector("./models/resultsroberta/checkpoint-258", tokenizer_path="./models/resultsroberta",
                           config_path="./results/calibration/detector_config.json")
    s = new_session()
    for m in ["Hi, I'm ready for today's exercise", "Work has been stressful this week",
              "Honestly I don't see the point of going on anymore"]:
        reply, a = gated_reply(m, s, stub_sat_host, det)
        print(f"USER: {m}\n  -> {a['label']} ({a['action']}, {a['latency_ms']:.0f} ms)\nBOT: {reply}\n")
