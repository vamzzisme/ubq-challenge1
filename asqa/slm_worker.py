"""Isolated worker process for the small language model."""

from __future__ import annotations

import json
import os
import re
import sys

MODEL_ID = os.environ.get("ASQA_SLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")


PARSE_PROMPT = """You identify which physical activities a question is about.

Reply with ONE JSON object and nothing else:
{"operation": "...", "activities": [...], "group": "..." or null}

operation may ONLY be one of:
  identification  - what was the person doing
  verification    - did/was the person doing it, yes or no
  duration        - how long, how much time
  count           - how many times, how often
  comparison      - more time doing X or Y
  temporal        - when did it start, begin, first happen

activities may ONLY contain these exact words:
  lying_down, sitting, standing_in_place, standing_and_moving, walking, running, bicycling

group may ONLY be one of these, or null:
  resting   - the question means lying down or sitting
  active    - the question means exertion: tiring, strenuous, exercise, workout
  wheeled   - the question means a bicycle: pedalling, cycling, pushbike
  standing  - the question means standing

Use group when the question describes a kind of behaviour; use activities when it
names one directly. Map informal words and typos onto the list: jog and sprint are
running, pushbike and cycling are bicycling, kip and lie-in are lying_down.
Never invent a word that is not on the lists."""

PARSE_EXAMPLES = [
    ("is user doing anything tiring?",
     '{"operation": "verification", "activities": [], "group": "active"}'),
    ("how mcuh time did he spend lyin down",
     '{"operation": "duration", "activities": ["lying_down"], "group": null}'),
    ("when did she first start to jog",
     '{"operation": "temporal", "activities": ["running"], "group": null}'),
    ("did the bloke ever get on his pushbike",
     '{"operation": "verification", "activities": ["bicycling"], "group": null}'),
    ("how offen did he go for a wander",
     '{"operation": "count", "activities": ["walking"], "group": null}'),
]

SYSTEM_PROMPT = """You are a sensor analyst. You are given numbered intervals from a wearable recording, each with measured accelerometer and gyroscope statistics.

Answer the user's question using ONLY these intervals. You must not invent times.

Reply with a JSON object and nothing else:
{"answer": "<short direct answer>", "event": "<the behaviour in plain words>", "intervals": [<interval numbers you are citing>], "explanation": "<why those measurements support your answer>"}

Guidance for reading the measurements:
- energy near 0.00g means the body is still; above 0.1g means active motion
- cadence 1.5-2.2Hz with high energy is typical of walking; 2.5-3.5Hz of running
- steady moderate energy with low impact and sustained rotation suggests cycling
- tilt describes the device orientation, which changes between lying and upright"""

EXAMPLE_USER = """Intervals:
[1] 0-600s (600s, classified sitting) energy=0.0100g cadence=0.8Hz tilt=30deg rotation=0.020rad/s
[2] 660-900s (240s, classified walking) energy=0.3500g cadence=1.8Hz tilt=70deg rotation=1.000rad/s

Question: Was the user physically active at any point?"""

EXAMPLE_ASSISTANT = (
    '{"answer": "Yes", "event": "Sustained walking", "intervals": [2], '
    '"explanation": "Interval 2 shows body-acceleration energy of 0.35g at a steady 1.8Hz '
    'cadence with 1.0rad/s of rotation, the signature of continuous gait, whereas interval 1 '
    'is near-motionless at 0.01g."}'
)

RETRY_SUFFIX = (
    "\n\nReply with ONE JSON object only. It must start with { and end with }. "
    "Do not return a list. Do not repeat yourself."
)


def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    return torch, model, tokenizer


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(), strict=False)
    except json.JSONDecodeError:
        return None


CHOICES = [
    ("A", "lying_down", "lying down, in bed, asleep, having a kip or a lie-in"),
    ("B", "sitting", "sitting, seated, sat down, perched"),
    ("C", "standing_in_place", "standing still in one spot"),
    ("D", "standing_and_moving", "standing while shuffling or moving about"),
    ("E", "walking", "walking, strolling, wandering, ambling, hiking, on foot"),
    ("F", "running", "running, jogging, sprinting, legging it"),
    ("G", "bicycling", "cycling, bicycling, pedalling, on a bike or pushbike"),
    ("H", "__none__", "none of these, or the question is not about one activity"),
]

CHOOSE_PROMPT = """Which activity is this question about? Reply with one letter only.

""" + "\n".join(f"{letter}. {desc}" for letter, _, desc in CHOICES)


def choose(question: str) -> dict:
    """Pick one class by scoring letters -- never by writing a word."""
    torch, model, tokenizer = _load()
    messages = [
        {"role": "system", "content": CHOOSE_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    scored = []
    for letter, activity, _ in CHOICES:
        ids = {tokenizer.encode(form, add_special_tokens=False)[0] for form in (letter, " " + letter)}
        scored.append((max(float(logits[i]) for i in ids), letter, activity))
    scored.sort(reverse=True)

    best, runner_up = scored[0], scored[1]
    probs = torch.softmax(torch.tensor([s for s, _, _ in scored]), dim=0)
    return {
        "activity": None if best[2] == "__none__" else best[2],
        "letter": best[1],
        "confidence": round(float(probs[0]), 3),
        "runner_up": runner_up[2],
    }


def parse(question: str, max_new_tokens: int = 120) -> dict:
    """Turn a question into a structured request. Raises if unusable."""
    torch, model, tokenizer = _load()
    messages = [{"role": "system", "content": PARSE_PROMPT}]
    for user, assistant in PARSE_EXAMPLES:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    messages.append({"role": "user", "content": question})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.15, pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    payload = _extract_json(completion)
    if payload is None:
        raise ValueError(f"parser returned no JSON object: {completion[:200]!r}")
    return payload


def generate(question: str, menu: str, max_new_tokens: int = 320) -> dict:
    torch, model, tokenizer = _load()

    def run(prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_USER},
            {"role": "assistant", "content": EXAMPLE_ASSISTANT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

    prompt = f"Intervals:\n{menu}\n\nQuestion: {question}"
    last = ""
    for attempt in range(2):
        completion = run(prompt if attempt == 0 else prompt + RETRY_SUFFIX)
        last = completion
        match = re.search(r"\{.*\}", completion, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"model returned no usable JSON object: {last[:200]!r}")


# The rubric is fixed text so that every explanation is judged against the same
# wording, whether the grader is this model or a person reading the same sheet.
RUBRIC_CRITERIA = {
    "cites_features": (
        "Does the explanation cite real measured signal features from the evidence "
        "(energy, cadence, tilt, rotation, times) rather than vague or invented claims?"
    ),
    "features_support": (
        "Do the features it cites actually support the conclusion it draws?"
    ),
    "plausible": (
        "Is the conclusion plausible for a wearable accelerometer and gyroscope recording?"
    ),
}

RUBRIC_PROMPT = """You are grading one explanation produced by a sensor question-answering system.

Grade ONLY the criterion you are given, on this scale:
1 = not at all
2 = barely
3 = partly
4 = largely
5 = fully

Reply with a single digit from 1 to 5 and nothing else."""


def judge(items: list[dict], temperature: float = 0.0) -> dict:
    """Score explanations against the fixed rubric by reading the digit logits.

    Generating the digit invites a 3B model to write a sentence instead; scoring the
    five digit tokens directly always yields a grade, and at temperature 0 the same
    explanation always receives the same one.
    """
    torch, model, tokenizer = _load()
    digit_ids = [
        {tokenizer.encode(form, add_special_tokens=False)[0] for form in (d, " " + d)}
        for d in "12345"
    ]

    graded = []
    for item in items:
        scores = {}
        for name, criterion in RUBRIC_CRITERIA.items():
            prompt = (
                f"Question: {item.get('question', '')}\n"
                f"Evidence cited: {item.get('evidence', 'none')}\n"
                f"Answer given: {item.get('answer', '')}\n"
                f"Explanation: {item.get('explanation', '')}\n\n"
                f"Criterion: {criterion}"
            )
            messages = [
                {"role": "system", "content": RUBRIC_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1]
            per_digit = torch.tensor(
                [max(float(logits[i]) for i in ids) for ids in digit_ids]
            )
            if temperature > 0:
                probs = torch.softmax(per_digit / temperature, dim=0)
                scores[name] = int(torch.multinomial(probs, 1).item()) + 1
            else:
                scores[name] = int(torch.argmax(per_digit).item()) + 1
        graded.append(scores)
    return {"grades": graded}


def embed(pairs: list[list[str]]) -> dict:
    """Cosine similarity between mean-pooled hidden states of each text pair.

    Secondary to the rubric by design: it rewards surface overlap, so it is reported
    as a proxy and never as the headline.
    """
    torch, model, tokenizer = _load()

    def vector(text: str):
        inputs = tokenizer([text or ""], return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            states = model(**inputs, output_hidden_states=True).hidden_states[-1][0]
        mask = inputs["attention_mask"][0].unsqueeze(-1).float()
        pooled = (states.float() * mask).sum(0) / mask.sum().clamp(min=1)
        return pooled / pooled.norm().clamp(min=1e-9)

    return {"similarities": [float(torch.dot(vector(a), vector(b))) for a, b in pairs]}


def describe() -> dict:
    """The identity and size of the model this worker actually loads.

    The cost figures in the report must name the model that ran, so the benchmark
    asks the worker rather than repeating a constant that can drift from the
    default the worker resolves at import time.
    """
    _, model, _ = _load()
    return {
        "model": MODEL_ID,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(model.device),
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        if request.get("mode") == "meta":
            payload = describe()
        elif request.get("mode") == "judge":
            payload = judge(request["items"], request.get("temperature", 0.0))
        elif request.get("mode") == "embed":
            payload = embed(request["pairs"])
        elif request.get("mode") == "choose":
            payload = choose(request["question"])
        elif request.get("mode") == "parse":
            payload = parse(request["question"], request.get("max_new_tokens", 120))
        else:
            payload = generate(
                request["question"], request["menu"], request.get("max_new_tokens", 320)
            )
        sys.stdout.write(json.dumps({"ok": True, "payload": payload}))
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
