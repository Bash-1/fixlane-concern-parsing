"""Shared constants, prompts and helpers for training, baselines and evaluation."""
from __future__ import annotations
import json, re

SYSTEMS = ["engine", "electrical", "brakes", "suspension", "hvac", "drivetrain", "body", "other"]
SEVERITIES = ["safety_critical", "drivability", "comfort", "cosmetic"]
FIELDS = ["vehicle_system", "primary_symptom", "severity", "suggested_diagnostic"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_target(output: dict) -> str:
    """Canonical JSON string for a label (fixed key order). Used for few-shot turns and fine-tune targets."""
    return json.dumps({k: output[k] for k in FIELDS}, ensure_ascii=False)


# ---- Text normalization used for retrieval / dedup (strips filler wrappers, keeps the concern) ----
FILLER = ["tech notes", "note", "verified that", "on inspection", "during road test", "observed",
          "reproduced", "per customer", "owner says", "customer states", "customer state", "cust states",
          "cust state", "customer reports", "cust reports", "customer complaint", "cust complaint",
          "cust complain", "customer says", "cust says", "not always", "needs attention",
          "scheduled f/u", "scheduled fu", "if possible", "customer waiting", "cust waiting",
          "please advise", "urgent", "intermittent"]
_FILLER_RE = [re.compile(rf"\b{re.escape(f)}\b") for f in FILLER]

def core_text(s: str) -> str:
    s = re.sub(r"[^a-z0-9/ ]", " ", s.lower())
    for rx in _FILLER_RE:
        s = rx.sub(" ", s)
    return " ".join(s.split())


# ---- Prompt for the fine-tuned model: short, because conventions are learned from data ----
SYSTEM_PROMPT_FT = (
    "Parse the vehicle repair concern into JSON with keys vehicle_system "
    f"({', '.join(SYSTEMS)}), primary_symptom (short noun phrase), severity "
    f"({', '.join(SEVERITIES)}), suggested_diagnostic (short imperative phrase or null). "
    "Return only the JSON object."
)

# ---- Prompt for the prompted baseline: full instructions + Fixlane conventions from the train audit ----
SYSTEM_PROMPT_BASELINE = """You parse free-form vehicle repair concerns written by Fixlane technicians and service advisors into structured fields. Inputs contain typos, abbreviations (cust, veh, f/u, CEL, SOC) and filler ("cust states", "needs attention", "customer waiting") that carry no meaning. Many vehicles are EVs.

Return a JSON object with exactly these keys:
- vehicle_system: one of engine, electrical, brakes, suspension, hvac, drivetrain, body, other
- primary_symptom: short noun phrase (2-7 words) naming the main symptom plus its key context (where / when it happens)
- severity: one of safety_critical, drivability, comfort, cosmetic
- suggested_diagnostic: short imperative phrase (3-14 words) giving the first diagnostic step, or null if none applies

Severity definitions:
- safety_critical: affects ability to safely operate the vehicle
- drivability: affects normal use but not safety
- comfort: affects convenience or pleasure of driving
- cosmetic: appearance only

Fixlane labeling conventions (follow these over general intuition):
vehicle_system
- EV high-voltage battery, range, battery state of health, regenerative braking, drive unit/motor, and reduced-power behavior -> drivetrain
- 12V starting/charging, lighting, cameras, window motors, power steering assist (EPS), vehicle wake/sleep, shifter/park control, control modules -> electrical
- Infotainment software and updates, security alarm, TPMS -> other
- Closures, latches, sunroof, trunk/frunk, trim, water leaks -> body
severity
- safety_critical is used narrowly. Steering wheel shake, pulling on a flat road, bouncing, defroster performance, backup camera faults, a parking brake that won't release, and a door that needs slamming to latch are drivability.
- A feature that is unavailable and blocks normal use (frunk won't open, infotainment unusable, water entering the cabin) is drivability.
- Perceived performance changes with no fault (slower than when new, sluggish only in eco mode, lower mpg) and TPMS light after inflation are comfort.
- cosmetic is appearance only; anything that makes noise or stops working is not cosmetic.

Match the style of the examples: concise, specific, technician language."""

# ---- JSON schema for API structured outputs ----
OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "vehicle_system": {"type": "string", "enum": SYSTEMS},
        "primary_symptom": {"type": "string"},
        "severity": {"type": "string", "enum": SEVERITIES},
        "suggested_diagnostic": {"type": ["string", "null"]},
    },
    "required": FIELDS,
    "additionalProperties": False,
}
