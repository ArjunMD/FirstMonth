# app.py
# Run: streamlit run app.py

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

# =============================================================================
# Constants / Conversions
# =============================================================================
LB_TO_G = 453.59237
OZ_TO_G = 28.349523125
KG_TO_G = 1000.0

# For breast milk / formula, cc == mL, and 1 US fl oz = 29.5735295625 mL
ML_PER_FL_OZ = 29.5735295625

# Subtracted from diaper-only weights for plotting (fallback if event doesn't store tare)
DIAPER_TARE_G = 17.0

DATA_DIR = Path("data")
EVENTS_FILE = DATA_DIR / "events.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
THANKS_FILE = DATA_DIR / "thank_you.json"
ADVICE_FILE = DATA_DIR / "advice.json"
GROCERIES_FILE = DATA_DIR / "groceries.json"

ADVICE_CATEGORIES = [
    "General",
    "Feeding",
    "Sleep",
    "Diapering",
    "Soothing",
    "Health/Safety",
    "Gear",
    "Postpartum",
    "Relationships",
]

ADVICE_STATUSES = ["Untried", "Worked", "Sort of worked", "Doesn't work"]

GROCERY_STORES = [
    "Grocery/Any",
    "Trader Joes",
    "Costco",
    "Amazon",
    "Target",
    "Other",
]

THANKS_RESPONSE_OPTIONS = ["(blank)", "text", "thank you note"]
PRE_COND_OPTIONS = ["(blank)", "naked", "clean diaper", "clothed", "other"]
VITD_EVENT_TYPE = "vitamin_d"


# =============================================================================
# Weight parsing helpers
# =============================================================================
UNIT_ALIASES = {
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
}

PATTERN = re.compile(
    r"([-+]?\d*\.?\d+)\s*(kg|kilograms?|g|grams?|lb|lbs|pounds?|oz|ounces?)\b", re.I
)


def parse_weight_to_grams(text: str) -> float | None:
    """
    Parse free-text weight inputs into grams.

    Examples:
      - "3245 g"
      - "3.245 kg"
      - "6 lb 11.3 oz"
      - "7lbs"
      - "110 oz"

    Returns grams, or None if the input can't be parsed.
    """
    if not text or not text.strip():
        return None

    s = text.strip().lower().replace(",", " ")
    matches = PATTERN.findall(s)
    if not matches:
        return None

    totals = {"g": 0.0, "kg": 0.0, "lb": 0.0, "oz": 0.0}
    for num_str, unit_raw in matches:
        unit = UNIT_ALIASES.get(unit_raw.lower())
        if not unit:
            continue
        try:
            val = float(num_str)
        except ValueError:
            continue
        totals[unit] += val

    grams = (
        totals["g"]
        + totals["kg"] * KG_TO_G
        + totals["lb"] * LB_TO_G
        + totals["oz"] * OZ_TO_G
    )

    return grams if grams > 0 else None


GRAMS_ONLY_PATTERN = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*g?\s*$", re.I)


def parse_grams_only(text: str) -> float | None:
    """
    Accepts: "3245", "3245 g", "3245g"
    Rejects: "7 lb", "3.2 kg", "12 oz", etc.
    Returns grams, or None.
    """
    if not text or not text.strip():
        return None
    m = GRAMS_ONLY_PATTERN.match(text.strip())
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if v > 0 else None


NUMBER_ONLY_PATTERN = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*$")


def parse_number_only(text: str) -> float | None:
    """
    Accepts: "3245", "32", "32.5"
    Rejects: anything containing units/letters like "3245 g", "7 lb", "12oz"
    Returns positive float, or None.
    """
    if not text or not text.strip():
        return None
    m = NUMBER_ONLY_PATTERN.match(text.strip())
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if v > 0 else None


def fmt_g(v: Optional[float]) -> str:
    """Format a grams value for inputs (no trailing .0)."""
    if v is None:
        return ""
    return f"{float(v):g}"


def grams_to_lbs_oz(g: float) -> tuple[int, float]:
    total_oz = g / OZ_TO_G
    lbs = int(total_oz // 16)
    oz = total_oz - (lbs * 16)
    oz = round(oz, 1)
    if oz >= 16.0:
        lbs += 1
        oz = 0.0
    return lbs, oz


def ml_to_floz(ml: float) -> float:
    return ml / ML_PER_FL_OZ


def floz_to_ml(floz: float) -> float:
    return floz * ML_PER_FL_OZ


# =============================================================================
# Storage helpers
# =============================================================================
def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    _ensure_data_dir()
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(settings: Dict[str, Any]) -> None:
    _ensure_data_dir()
    SETTINGS_FILE.write_text(
        json.dumps(settings, indent=2, sort_keys=False), encoding="utf-8"
    )


def _settings_cache() -> Dict[str, Any]:
    if "_settings_cache" not in st.session_state:
        st.session_state["_settings_cache"] = load_settings()
    return st.session_state["_settings_cache"]


def persist_settings(patch: Dict[str, Any]) -> None:
    s = _settings_cache()
    s.update(patch)
    save_settings(s)


def load_events() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if not EVENTS_FILE.exists():
        return []
    try:
        data = json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_events(events: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    EVENTS_FILE.write_text(json.dumps(events, indent=2, sort_keys=False), encoding="utf-8")


def add_event(events: List[Dict[str, Any]], event: Dict[str, Any]) -> List[Dict[str, Any]]:
    events.append(event)
    events.sort(key=lambda e: e.get("ts", ""))
    save_events(events)
    return events


def update_event(events: List[Dict[str, Any]], event_id: str, patch: Dict[str, Any]) -> List[Dict[str, Any]]:
    for i, e in enumerate(events):
        if e.get("id") == event_id:
            events[i] = {**e, **patch}
            break
    events.sort(key=lambda e: e.get("ts", ""))
    save_events(events)
    return events


def delete_event(events: List[Dict[str, Any]], event_id: str) -> List[Dict[str, Any]]:
    events = [e for e in events if e.get("id") != event_id]
    save_events(events)
    return events


def load_thank_yous() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if not THANKS_FILE.exists():
        return []
    try:
        data = json.loads(THANKS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_thank_yous(items: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    THANKS_FILE.write_text(json.dumps(items, indent=2, sort_keys=False), encoding="utf-8")


def load_advice() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if not ADVICE_FILE.exists():
        return []
    try:
        data = json.loads(ADVICE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_advice(items: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    ADVICE_FILE.write_text(json.dumps(items, indent=2, sort_keys=False), encoding="utf-8")


def add_advice(items: List[Dict[str, Any]], item: Dict[str, Any]) -> List[Dict[str, Any]]:
    items.append(item)
    save_advice(items)
    return items


def update_advice(items: List[Dict[str, Any]], item_id: str, patch: Dict[str, Any]) -> List[Dict[str, Any]]:
    for i, it in enumerate(items):
        if it.get("id") == item_id:
            items[i] = {**it, **patch}
            break
    save_advice(items)
    return items


def delete_advice(items: List[Dict[str, Any]], item_id: str) -> List[Dict[str, Any]]:
    items = [it for it in items if it.get("id") != item_id]
    save_advice(items)
    return items


def load_groceries() -> List[Dict[str, Any]]:
    """
    Load groceries and normalize the schema.

    Normalized schema:
      - id: str
      - created_ts: iso str
      - text: str (no leading stars)
      - store: str (one of GROCERY_STORES)
      - done: bool
      - priority: bool
    """
    _ensure_data_dir()
    if not GROCERIES_FILE.exists():
        return []
    try:
        items = json.loads(GROCERIES_FILE.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            return []
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    migrated = False

    for it in items:
        if not isinstance(it, dict):
            continue

        raw_text = str(it.get("text") or "").strip()

        # Infer priority from legacy leading stars, and strip them from stored text
        had_star = bool(re.match(r"^\s*(?:[⭐☆]\s*)+", raw_text))
        priority = bool(it.get("priority", False)) or had_star
        text = re.sub(r"^\s*(?:[⭐☆]\s*)+", "", raw_text).strip()

        if not text:
            migrated = True
            continue

        store = str(it.get("store") or "Grocery/Any").strip() or "Grocery/Any"
        if store not in GROCERY_STORES:
            store = "Other"
            migrated = True

        done = bool(it.get("done", False))
        created_ts = str(it.get("created_ts") or dt_to_iso(datetime.now()))
        rid = str(it.get("id") or uuid.uuid4())

        if (
            raw_text != text
            or had_star
            or ("priority" not in it)
            or ("created_ts" not in it)
            or ("id" not in it)
        ):
            migrated = True

        out.append(
            {
                "id": rid,
                "created_ts": created_ts,
                "text": text,
                "store": store,
                "done": done,
                "priority": priority,
            }
        )

    if migrated:
        save_groceries(out)

    return out


def save_groceries(items: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    GROCERIES_FILE.write_text(json.dumps(items, indent=2, sort_keys=False), encoding="utf-8")


# =============================================================================
# Datetime helpers
# =============================================================================
def dt_to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def compose_dt(d: date, t: time) -> datetime:
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)


def format_snapshot_dt(dt: datetime) -> str:
    return f"{dt.strftime('%b')} {dt.day}, {dt.strftime('%H:%M')}"


def time_picker(
    label: str,
    default: time,
    *,
    key: str,
    show_quick: bool = True,
) -> time:
    """
    Native time input + quick back buttons.
    Avoids Streamlit warning by NOT passing value= once session_state has the key.
    """
    widget_key = f"{key}__ti"

    # Initialize the widget's state BEFORE creating the widget
    if widget_key not in st.session_state:
        st.session_state[widget_key] = default.replace(second=0, microsecond=0)

    def _shift_minutes(delta_min: int) -> None:
        cur: time = st.session_state.get(widget_key) or default
        base = datetime.combine(date.today(), cur)
        st.session_state[widget_key] = (base + timedelta(minutes=int(delta_min))).time().replace(
            second=0, microsecond=0
        )

    picked: time = st.time_input(label, key=widget_key)
    picked = picked.replace(second=0, microsecond=0)

    if show_quick:
        def _set_now() -> None:
            st.session_state[widget_key] = datetime.now().time().replace(second=0, microsecond=0)

        # One row, only 3 buttons -> wide enough to avoid ugly wrapping
        c1, c2, c3 = st.columns([1, 1, 1], gap="small")

        c1.button("Now",  key=f"{key}__now", on_click=_set_now, use_container_width=True)
        c2.button("−5m",  key=f"{key}__m5",  on_click=_shift_minutes, args=(-5,),  use_container_width=True)
        c3.button("−10m", key=f"{key}__m10", on_click=_shift_minutes, args=(-10,), use_container_width=True)

        picked = (st.session_state.get(widget_key) or picked).replace(second=0, microsecond=0)

    return picked


# =============================================================================
# Weight extraction for plotting
# =============================================================================
def adjusted_weight_if_plotworthy(
    weight_g: Optional[float],
    condition: Optional[str],
    diaper_tare_g: Optional[float] = None,
) -> Optional[float]:
    """
    Only plot naked or diaper-only weights.
    For diaper-only ("clean diaper"), subtract diaper_tare_g if provided,
    otherwise fall back to DIAPER_TARE_G.
    """
    if weight_g is None:
        return None

    cond = (condition or "").strip().lower()
    if cond not in {"naked", "clean diaper"}:
        return None

    if cond == "clean diaper":
        tare = DIAPER_TARE_G
        if diaper_tare_g is not None:
            try:
                tare = float(diaper_tare_g)
            except Exception:
                tare = DIAPER_TARE_G
        if tare <= 0:
            tare = DIAPER_TARE_G
        return float(weight_g) - float(tare)

    return float(weight_g)


def weight_for_event_row(e: Dict[str, Any]) -> Optional[float]:
    """
    Only random weight events are plotted (naked or clean-diaper).
    """
    if e.get("type") != "weight":
        return None

    return adjusted_weight_if_plotworthy(
        e.get("weight_g"),
        e.get("condition"),
        e.get("diaper_tare_g"),
    )


def recent_naked_weights(events: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    """
    Return up to n most recent naked weights with timestamps.
    """
    out: List[Dict[str, Any]] = []

    for e in sorted(events, key=lambda x: x.get("ts", ""), reverse=True):
        ts = e.get("ts")
        if not ts or e.get("type") != "weight":
            continue

        try:
            dt = iso_to_dt(ts)
        except Exception:
            continue

        if (e.get("condition") or "").strip().lower() != "naked":
            continue

        if e.get("weight_g") is None:
            continue

        out.append({"dt": dt, "weight_g": float(e["weight_g"]), "source": "weight"})
        if len(out) >= n:
            break

    return out


def vitamin_d_given_dt(events: List[Dict[str, Any]], day: date) -> Optional[datetime]:
    """Return the first Vitamin D timestamp for the given day, or None."""
    for e in sorted(events, key=lambda x: x.get("ts", "")):  # oldest first
        if e.get("type") != VITD_EVENT_TYPE:
            continue
        ts = e.get("ts")
        if not ts:
            continue
        try:
            dt = iso_to_dt(ts)
        except Exception:
            continue
        if dt.date() == day:
            return dt
    return None


# =============================================================================
# Feeding / breastfeeding helpers
# =============================================================================
def feeding_intake_ml(e: Dict[str, Any]) -> Optional[float]:
    """
    Return intake for feeding events in mL.

    - Bottle: bottle_ml
    - Breast: delta_g (treated as mL)

    Legacy support: if delta_g missing, falls back to (post-pre) weights if present.
    Assumption: 1 g ~= 1 mL.
    """
    if e.get("type") != "feeding":
        return None

    mode = e.get("feed_mode")
    if mode == "bottle":
        ml = e.get("bottle_ml")
        return float(ml) if ml else None

    # Breast (preferred)
    dg = e.get("delta_g")
    if dg is not None:
        dg = float(dg or 0.0)
        return dg if dg > 0 else None

    # Legacy fallback (old events may have used pre/post, but unsure how many actually did)
    pre = e.get("pre_weight_g")
    post = e.get("post_weight_g")
    if pre is None or post is None:
        return None

    delta = float(post) - float(pre)
    return delta if delta > 0 else None



def _toggle_side(side: str) -> str:
    return "R" if side == "L" else "L"


def breast_side_status(events: List[Dict[str, Any]]) -> tuple[Optional[str], str]:
    """
    Return (last_start_side, next_start_side) for breastfeeding events.

    - Scans feeding events with feed_mode == "breast" in chronological order.
    - If an older event is missing start_side, infer by alternating,
      starting from "L" if no prior side is known.
    - next_start_side is the opposite of the last inferred/known side,
      or "L" if there have been no breastfeeds.
    """
    breast_events = [
        e
        for e in sorted(events, key=lambda x: x.get("ts", ""))
        if e.get("type") == "feeding" and e.get("feed_mode") == "breast"
    ]

    last: Optional[str] = None
    seen_any = False

    for e in breast_events:
        seen_any = True
        side = (e.get("start_side") or "").upper().strip()
        if side in {"L", "R"}:
            last = side
        else:
            last = "L" if last is None else _toggle_side(last)

    if not seen_any:
        return None, "L"

    assert last in {"L", "R"}
    return last, _toggle_side(last)


# =============================================================================
# UI helpers
# =============================================================================
def flash(msg: str, icon: str = "✅") -> None:
    """Queue a toast message to be shown on the next rerun (works inside callbacks)."""
    st.session_state["_flash"] = (msg, icon)


def flash_and_rerun(msg: str, icon: str = "✅") -> None:
    """Use for buttons/forms (not callbacks)."""
    flash(msg, icon)
    st.rerun()


def status_checkbox_group(*, key_base: str, current: str = "Untried") -> str:
    """Render a mutually-exclusive checkbox group for ADVICE_STATUSES."""
    cur = current if current in ADVICE_STATUSES else "Untried"
    st.session_state.setdefault(f"{key_base}__status", cur)

    def _pick(chosen: str) -> None:
        st.session_state[f"{key_base}__status"] = chosen
        for opt in ADVICE_STATUSES:
            st.session_state[f"{key_base}__cb__{opt}"] = (opt == chosen)

    cols = st.columns(4)
    for col, opt in zip(cols, ADVICE_STATUSES):
        k = f"{key_base}__cb__{opt}"
        if k not in st.session_state:
            st.session_state[k] = (st.session_state[f"{key_base}__status"] == opt)
        col.checkbox(opt, key=k, on_change=_pick, args=(opt,))

    return st.session_state[f"{key_base}__status"]

def pretty_event_label(e: Dict[str, Any]) -> str:
    """Label for an event (used in Recent events titles)."""
    et = str(e.get("type") or "").strip().lower()

    if et == "feeding":
        mode = (e.get("feed_mode") or "").strip().lower()
        base = "Breastfeed" if mode == "breast" else ("Bottle feed" if mode == "bottle" else "Feeding")

        ml = feeding_intake_ml(e)
        if ml is not None and float(ml) > 0:
            oz = ml_to_floz(float(ml))
            return f"{base} - {oz:.2f} oz"
        return base

    if et == "pumping":
        total_ml = e.get("pump_total_ml")
        if total_ml is None:
            left = e.get("pump_left_ml")
            right = e.get("pump_right_ml")
            if left is not None or right is not None:
                total_ml = float(left or 0.0) + float(right or 0.0)

        if total_ml is not None and float(total_ml) > 0:
            oz = ml_to_floz(float(total_ml))
            return f"Pumping - {oz:.2f} oz"
        return "Pumping"

    if et == "diaper":
        desc = e.get("descriptor") or []
        if isinstance(desc, list):
            desc = [str(x).strip() for x in desc if str(x).strip()]
        else:
            desc = []

        if desc:
            return "Diaper change - " + ", ".join(desc)
        return "Diaper change"

    if et == "weight":
        w = e.get("weight_g")
        cond = (e.get("condition") or "").strip()
        if w is not None:
            try:
                w0 = float(w)
                cond_part = f" ({cond})" if cond else ""
                return f"Weight - {w0:,.0f} g{cond_part}"
            except Exception:
                pass
        return "Weight"

    if et == VITD_EVENT_TYPE:
        return "Vitamin D"

    # fallback
    return (e.get("type") or "Event").replace("_", " ").title()


# =============================================================================
# Tab renderers
# =============================================================================
def render_snapshot_tab() -> None:
    events = load_events()
    st.subheader("Snapshot")

    # --- Last 3 naked weights ---
    with st.container(border=True):
        st.markdown("#### Last 3 naked weights")
        w = recent_naked_weights(events, n=3)
        if not w:
            st.info("No naked weights recorded yet.")
        else:
            rows = []
            for item in w:
                g = float(item["weight_g"])
                lbs, oz = grams_to_lbs_oz(g)
                rows.append(
                    {
                        "When": format_snapshot_dt(item["dt"]),
                        "Grams": f"{g:,.0f}",
                        "Lb/Oz": f"{lbs} lb {oz:.1f} oz",
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # --- Last breastfeeding side ---
    with st.container(border=True):
        st.markdown("#### Breastfeeding")
        last_side, next_side = breast_side_status(events)
        st.metric("Last start side", last_side or "—")
        st.caption(f"Next suggested start: {next_side}")

    # --- Vitamin D (once daily) ---
    with st.container(border=True):
        st.markdown("#### Vitamin D")
        today_vitd = vitamin_d_given_dt(events, date.today())

        if today_vitd is None:
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("✅ Yes", key="vitd_yes"):
                    e = {
                        "id": str(uuid.uuid4()),
                        "type": VITD_EVENT_TYPE,
                        "ts": dt_to_iso(datetime.now()),
                        "notes": "",
                    }
                    add_event(events, e)
                    st.rerun()
            with c2:
                st.caption("Not recorded yet today.")
        else:
            st.success(f"Given at {today_vitd.strftime('%H:%M')}")

    # --- Today's goal + intake + pace ---
    today = date.today()
    now_dt = datetime.now()
    midnight = datetime.combine(today, time(0, 0, 0))
    elapsed_hours = max((now_dt - midnight).total_seconds() / 3600.0, 1e-6)

    goal_floz_today: Optional[float] = None
    if st.session_state.get("goal_date") == str(today):
        g = st.session_state.get("goal_floz")
        goal_floz_today = float(g) if g is not None else None

    # Intake so far today
    today_total_ml = 0.0
    for e in events:
        ts = e.get("ts")
        if not ts:
            continue
        try:
            dt = iso_to_dt(ts)
        except Exception:
            continue
        if dt.date() != today:
            continue
        ml = feeding_intake_ml(e)
        if ml is None:
            continue
        today_total_ml += float(ml)

    today_total_floz = ml_to_floz(today_total_ml)

    # Previous two daily intakes
    def _day_intake_ml(day0: date) -> float:
        total = 0.0
        for e in events:
            ts = e.get("ts")
            if not ts:
                continue
            try:
                dt = iso_to_dt(ts)
            except Exception:
                continue
            if dt.date() != day0:
                continue
            ml = feeding_intake_ml(e)
            if ml is None:
                continue
            total += float(ml)
        return total

    prev_days = [today - timedelta(days=1), today - timedelta(days=2)]
    prev_rows = []
    for d0 in prev_days:
        ml0 = _day_intake_ml(d0)
        label = "Yesterday" if d0 == (today - timedelta(days=1)) else "2 days ago"
        prev_rows.append(
            {"Day": f"{label} ({d0.strftime('%b')} {d0.day})", "Intake (oz)": round(ml_to_floz(ml0), 2)}
        )

    with st.container(border=True):
        st.markdown("#### Today")

        if goal_floz_today is None or goal_floz_today <= 0:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Intake so far", f"{today_total_floz:.2f} oz")
                st.caption(f"{today_total_ml:,.0f} mL")
            with c2:
                st.metric("Current pace", f"{(today_total_floz / elapsed_hours):.2f} oz/hr")
                st.caption("Set a goal in the Calculator tab to show on-pace status.")
        else:
            remaining_floz = max(goal_floz_today - today_total_floz, 0.0)
            remaining_hours = max(24.0 - elapsed_hours, 1e-6)

            expected_by_now = goal_floz_today * (elapsed_hours / 24.0)
            delta = today_total_floz - expected_by_now  # + ahead, - behind
            pct = min(today_total_floz / goal_floz_today, 1.0)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Intake so far", f"{today_total_floz:.2f} oz")
                st.caption(f"{today_total_ml:,.0f} mL")
            with r2:
                st.metric("Goal", f"{goal_floz_today:.1f} oz")
                st.caption(f"≈ {floz_to_ml(goal_floz_today):,.0f} mL")
            with r3:
                st.metric("Remaining", f"{remaining_floz:.2f} oz")
                st.caption(f"Needed pace: {(remaining_floz / remaining_hours):.2f} oz/hr")

            st.progress(pct)

            if delta >= 0:
                st.caption(f"On pace for midnight (ahead by {delta:.2f} oz vs steady pace).")
            else:
                st.caption(f"Behind pace for midnight ({abs(delta):.2f} oz behind steady pace).")




            st.divider()
            st.markdown("##### Previous days")
            st.dataframe(pd.DataFrame(prev_rows), hide_index=True, width="stretch")

    # --- Priority groceries (undone priority only) ---
    groceries = load_groceries()
    pri_pending = [it for it in groceries if (not bool(it.get("done"))) and bool(it.get("priority", False))]
    pri_pending = sorted(pri_pending, key=lambda x: x.get("created_ts", ""))  # oldest first

    if pri_pending:
        with st.container(border=True):
            st.markdown("#### Priority groceries")
            max_show = 12
            show = pri_pending[:max_show]

            by_store: Dict[str, List[str]] = {}
            for it in show:
                store = str(it.get("store") or "Grocery/Any")
                text = str(it.get("text") or "").strip()
                if not text:
                    continue
                by_store.setdefault(store, []).append(text)

            for store in GROCERY_STORES:
                items_here = by_store.get(store) or []
                if items_here:
                    st.markdown(f"**{store}**: " + " • ".join(items_here))

            if len(pri_pending) > max_show:
                st.caption(f"…and {len(pri_pending) - max_show} more (see Groceries tab).")


def render_calculator_tab() -> None:
    st.caption(
        "Pick a recent naked weight, or type the baby's weight in any unit. "
        "Examples: 3245 g, 3.245 kg, 6 lb 11.3 oz, 7lbs"
    )

    events = load_events()
    recent = recent_naked_weights(events, n=10)

    def _save_calc_weight_text() -> None:
        persist_settings({"calc_weight_text": st.session_state.get("calc_weight_text", "")})

    def _save_calc_weight_source() -> None:
        persist_settings({"calc_weight_source": st.session_state.get("calc_weight_source", "Recent")})

    def _save_calc_recent_idx() -> None:
        persist_settings({"calc_recent_idx": st.session_state.get("calc_recent_idx", 0)})

    if not recent:
        st.session_state["calc_weight_source"] = "Custom"

    with st.container(border=True):
        st.subheader("Enter baby's weight")

        source = st.radio(
            "Weight source",
            ["Recent", "Custom"],
            horizontal=True,
            key="calc_weight_source",
            on_change=_save_calc_weight_source,
        )

        grams: Optional[float] = None

        if source == "Recent":
            st.caption("Showing recent **naked** weights (from random weight events).")

            labels: List[str] = []
            for item in recent:
                g = float(item["weight_g"])
                lbs, oz = grams_to_lbs_oz(g)
                labels.append(f"{format_snapshot_dt(item['dt'])} — {g:,.0f} g  ({lbs} lb {oz:.1f} oz)")

            idx = int(st.session_state.get("calc_recent_idx", 0) or 0)
            idx = max(0, min(idx, len(labels) - 1))
            if st.session_state.get("calc_recent_idx") != idx:
                st.session_state["calc_recent_idx"] = idx

            pick = st.selectbox(
                "Recent naked weight",
                options=list(range(len(labels))),
                key="calc_recent_idx",
                format_func=lambda i: labels[i],
                on_change=_save_calc_recent_idx,
            )

            grams = float(recent[int(pick)]["weight_g"])

        else:
            text = st.text_input(
                "Weight (free text)",
                placeholder="e.g. 6 lb 11.3 oz",
                help="Supported units: g, kg, lb/lbs, oz. Mixed units like '6 lb 11.3 oz' are OK.",
                key="calc_weight_text",
                on_change=_save_calc_weight_text,
            )
            grams = parse_weight_to_grams(text)

    if grams is None:
        st.info("Enter a weight above (like `3245 g` or `6 lb 11.3 oz`).")
        return

    kg = grams / KG_TO_G
    lbs_decimal = grams / LB_TO_G
    lbs_int, oz = grams_to_lbs_oz(grams)

    with st.container(border=True):
        st.subheader("Weight conversions")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Grams (g)", f"{grams:,.1f}")
            st.metric("Kilograms (kg)", f"{kg:.4f}")
        with c2:
            st.metric("Pounds (decimal)", f"{lbs_decimal:.4f} lb")
            st.metric("Pounds + ounces", f"{lbs_int} lb {oz:.1f} oz")

    with st.container(border=True):
        st.subheader("Daily intake estimates")

        daily_ml_150 = kg * 150.0
        daily_floz_150 = ml_to_floz(daily_ml_150)

        daily_floz_aap = lbs_decimal * 2.5
        daily_ml_aap = daily_floz_aap * ML_PER_FL_OZ

        st.markdown("### Formula 1: 150 mL per kg")
        a1, a2 = st.columns(2)
        with a1:
            st.metric("mL/day", f"{daily_ml_150:,.0f} mL")
        with a2:
            st.metric("oz/day", f"{daily_floz_150:,.2f} oz")

        st.divider()

        st.markdown("### Formula 2: 2.5 fl oz per lb")
        b1, b2 = st.columns(2)
        with b1:
            st.metric("mL/day", f"{daily_ml_aap:,.0f} mL")
        with b2:
            st.metric("oz/day", f"{daily_floz_aap:,.2f} oz")

    with st.container(border=True):
        st.subheader("Set today's feeding goal")

        lower_oz = float(min(daily_floz_150, daily_floz_aap))
        upper_oz = float(max(daily_floz_150, daily_floz_aap))

        start_oz = math.floor(lower_oz * 2) / 2
        if abs(start_oz - lower_oz) < 1e-9:
            start_oz -= 0.5

        end_oz = math.ceil(upper_oz * 2) / 2
        if abs(end_oz - upper_oz) < 1e-9:
            end_oz += 0.5

        steps = int(round((end_oz - start_oz) / 0.5))
        options = [round(start_oz + i * 0.5, 1) for i in range(steps + 1)]

        today_str = str(date.today())
        if st.session_state.get("goal_date") != today_str:
            st.session_state["goal_date"] = today_str
            persist_settings({"goal_date": today_str})

        def _nearest_half(x: float) -> float:
            return round(round(x * 2) / 2, 1)

        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("Use lower estimate", key="goal_pick_lower"):
                st.session_state["goal_floz"] = _nearest_half(lower_oz)
                st.session_state["goal_date"] = today_str
                persist_settings({"goal_floz": st.session_state["goal_floz"], "goal_date": today_str})
                st.rerun()
        with q2:
            if st.button("Use higher estimate", key="goal_pick_upper"):
                st.session_state["goal_floz"] = _nearest_half(upper_oz)
                st.session_state["goal_date"] = today_str
                persist_settings({"goal_floz": st.session_state["goal_floz"], "goal_date": today_str})
                st.rerun()
        with q3:
            if st.button("Clear goal", key="goal_clear"):
                st.session_state["goal_floz"] = None
                st.session_state["goal_date"] = today_str
                persist_settings({"goal_floz": None, "goal_date": today_str})
                st.rerun()

        st.caption(
            f"Buttons shown from **{start_oz:.1f} oz** (just below {lower_oz:.2f}) "
            f"to **{end_oz:.1f} oz** (just above {upper_oz:.2f}), in 0.5 oz steps."
        )

        goal = st.session_state.get("goal_floz")
        if goal is None:
            st.info("Pick a goal below.")
        else:
            g1, g2 = st.columns(2)
            with g1:
                st.metric("Today's goal", f"{goal:.1f} fl oz/day")
            with g2:
                st.metric("≈ in mL/day", f"{floz_to_ml(goal):,.0f} mL")

        per_row = 6
        for r in range(0, len(options), per_row):
            cols = st.columns(per_row)
            for j, opt in enumerate(options[r : r + per_row]):
                label = f"✅ {opt:.1f} oz" if goal == opt else f"{opt:.1f} oz"
                if cols[j].button(label, key=f"goal_btn_{opt:.1f}"):
                    st.session_state["goal_floz"] = opt
                    st.session_state["goal_date"] = today_str
                    persist_settings({"goal_floz": opt, "goal_date": today_str})
                    st.rerun()

    with st.expander("Notes / assumptions", expanded=False):
        st.write("- Conversion used: **1 US fl oz = 29.5735 mL**.")
        st.write(f"- Graph diaper tare: **{DIAPER_TARE_G:.0f} g** subtracted from diaper-only weights (Graph tab).")
        st.write("- Breastfeed intake estimate assumes **1 g ≈ 1 mL** for milk transfer when using weights/deltas.")


def render_event_tracker_tab() -> None:
    events = load_events()
    now = datetime.now()

    st.caption("Log diaper changes, feeds, and weights. Stored locally in `data/events.json`.")

    with st.container(border=True):
        st.subheader("Add an event")
        etype = st.radio(
            "Event type",
            ["Diaper change", "Feeding", "Pumping", "Random weight"],
            index=1,
            horizontal=True,
            key="event_type",
        )

        col_d, col_t = st.columns(2)
        with col_d:
            d = st.date_input("Date", value=now.date(), key="add_date")
        with col_t:
            t = time_picker("Time", default=now.time(), key="add_time", show_quick=True)

        ts = dt_to_iso(compose_dt(d, t))

        if etype == "Diaper change":
            with st.form("add_diaper_form", clear_on_submit=True):
                desc = st.multiselect(
                    "Descriptor (optional)",
                    options=["pee", "potty", "robust", "light"],
                    default=[],
                    help="Pick any that apply.",
                )
                notes = st.text_input("Notes (optional)", placeholder="Anything else you want to remember?")
                submitted = st.form_submit_button("Add diaper change")
                if submitted:
                    e = {
                        "id": str(uuid.uuid4()),
                        "type": "diaper",
                        "ts": ts,
                        "descriptor": desc,
                        "notes": notes.strip() if notes else "",
                    }
                    add_event(events, e)
                    flash_and_rerun("Added diaper change.")

        elif etype == "Feeding":
            mode = st.radio("Feeding type", ["Breastfeed", "Bottle feed"], horizontal=True, key="add_feed_mode")

            if mode == "Bottle feed":
                b_unit = st.radio(
                    "Bottle amount unit",
                    ["oz", "mL / g (≈ same)"],
                    index=0,
                    horizontal=True,
                    key="add_bottle_unit",
                )
                st.caption("If you choose mL / g: we assume **1 mL ≈ 1 g**, so enter either without conversions.")

                with st.form("add_bottle_form", clear_on_submit=True):
                    step = 0.5 if b_unit == "oz" else 5.0
                    label = "Amount (oz)" if b_unit == "oz" else "Amount (mL or g)"
                    amt = st.number_input(label, min_value=0.0, step=step, value=None, placeholder="e.g. 30")

                    if amt and amt > 0:
                        if b_unit == "oz":
                            amt_ml = floz_to_ml(float(amt))
                            st.caption(f"≈ {amt_ml:,.0f} mL (≈ g)")
                        else:
                            amt_ml = float(amt)
                            st.caption(f"≈ {ml_to_floz(amt_ml):.2f} oz")
                    else:
                        amt_ml = 0.0

                    notes = st.text_input("Notes (optional)", placeholder="Side, spit-up, etc.")
                    submitted = st.form_submit_button("Add feeding")
                    if submitted:
                        e = {
                            "id": str(uuid.uuid4()),
                            "type": "feeding",
                            "ts": ts,
                            "feed_mode": "bottle",
                            "bottle_ml": float(amt_ml) if amt_ml > 0 else None,
                            "delta_g": None,
                            "notes": notes.strip() if notes else "",
                            "pre_weight_g": None,
                            "pre_condition": None,
                            "post_weight_g": None,
                        }
                        add_event(events, e)
                        flash_and_rerun("Added bottle feed.")

            else:
                with st.form("add_breast_form", clear_on_submit=True):
                    _, suggested_next = breast_side_status(events)
                    start_side = st.radio(
                        "Starting side (required)",
                        ["L", "R"],
                        index=0 if suggested_next == "L" else 1,
                        horizontal=True,
                    )

                    delta_text = st.text_input(
                        "Delta (g) — enter a number only (no units). "
                        "If unweighted feed, enter a conservative estimate here.",
                        placeholder="e.g. 32",
                    )
                    delta_g = None
                    if delta_text.strip():
                        delta_g = parse_number_only(delta_text)
                        if delta_g is None:
                            st.warning("Delta must be a number in grams only (e.g., `32` or `32.5`). No units.")
                        else:
                            st.caption(f"≈ {ml_to_floz(delta_g):.2f} oz (assuming 1 g ≈ 1 mL)")

                    notes = st.text_input("Notes (optional)", placeholder="Latch, duration, etc.")
                    submitted = st.form_submit_button("Add feeding")
                    if submitted:
                        e = {
                            "id": str(uuid.uuid4()),
                            "type": "feeding",
                            "ts": ts,
                            "feed_mode": "breast",
                            "start_side": start_side,
                            "bottle_ml": None,
                            "delta_g": float(delta_g) if (delta_g is not None and delta_g > 0) else None,
                            "notes": notes.strip() if notes else "",
                        }
                        add_event(events, e)
                        flash_and_rerun("Added breastfeed.")


        elif etype == "Pumping":
            p_unit = st.radio(
                "Pumped amount unit",
                ["oz", "mL / g (≈ same)"],
                index=0,
                horizontal=True,
                key="add_pump_unit",
            )
            st.caption("If you choose mL / g: we assume **1 mL ≈ 1 g**, so enter either without conversions.")

            with st.form("add_pumping_form", clear_on_submit=True):
                step = 0.5 if p_unit == "oz" else 5.0
                total_label = "Total (oz)" if p_unit == "oz" else "Total (mL or g)"
                left_label = "Left (oz)" if p_unit == "oz" else "Left (mL or g)"
                right_label = "Right (oz)" if p_unit == "oz" else "Right (mL or g)"

                total = st.number_input(total_label, min_value=0.0, step=step, value=None, placeholder="e.g. 4")
                left = st.number_input(left_label, min_value=0.0, step=step, value=None, placeholder="e.g. 2")
                right = st.number_input(right_label, min_value=0.0, step=step, value=None, placeholder="e.g. 2")

                def _to_ml(v: float) -> float:
                    return floz_to_ml(v) if p_unit == "oz" else v

                if total is not None and total > 0:
                    st.caption(
                        f"Total ≈ {_to_ml(float(total)):.0f} mL"
                        if p_unit == "oz"
                        else f"Total ≈ {ml_to_floz(float(total)):.2f} oz"
                    )
                if left is not None and left > 0:
                    st.caption(
                        f"Left ≈ {_to_ml(float(left)):.0f} mL"
                        if p_unit == "oz"
                        else f"Left ≈ {ml_to_floz(float(left)):.2f} oz"
                    )
                if right is not None and right > 0:
                    st.caption(
                        f"Right ≈ {_to_ml(float(right)):.0f} mL"
                        if p_unit == "oz"
                        else f"Right ≈ {ml_to_floz(float(right)):.2f} oz"
                    )

                submitted = st.form_submit_button("Add pumping")
                if submitted:
                    total_v = float(total) if (total is not None and total > 0) else None
                    left_v = float(left) if (left is not None and left > 0) else None
                    right_v = float(right) if (right is not None and right > 0) else None

                    total_ml = floz_to_ml(total_v) if (p_unit == "oz" and total_v is not None) else total_v
                    left_ml = floz_to_ml(left_v) if (p_unit == "oz" and left_v is not None) else left_v
                    right_ml = floz_to_ml(right_v) if (p_unit == "oz" and right_v is not None) else right_v

                    if total_ml is None and (left_ml is not None or right_ml is not None):
                        total_ml = (left_ml or 0.0) + (right_ml or 0.0)
                        if total_ml <= 0:
                            total_ml = None

                    e = {
                        "id": str(uuid.uuid4()),
                        "type": "pumping",
                        "ts": ts,
                        "pump_total_ml": float(total_ml) if (total_ml is not None and total_ml > 0) else None,
                        "pump_left_ml": float(left_ml) if (left_ml is not None and left_ml > 0) else None,
                        "pump_right_ml": float(right_ml) if (right_ml is not None and right_ml > 0) else None,
                        "notes": "",
                    }
                    add_event(events, e)
                    flash_and_rerun("Added pumping.")

        else:
            condition = st.radio("Condition", ["naked", "clean diaper"], horizontal=True, key="add_weight_condition")

            diaper_tare_g: Optional[float] = None
            if condition == "clean diaper":
                st.session_state.setdefault(
                    "add_weight_diaper_tare_g",
                    float(st.session_state.get("last_diaper_tare_g", DIAPER_TARE_G)),
                )
                diaper_tare_g = st.number_input(
                    "Diaper tare (g) for this event",
                    min_value=0.0,
                    step=1.0,
                    key="add_weight_diaper_tare_g",
                    help="Stored on this weight event (used when weight is diaper-only). Defaults to your last-used tare.",
                )

            with st.form("add_weight_form", clear_on_submit=True):
                w_text = st.text_input("Weight (g)", placeholder="e.g. 3245")
                w_g = parse_number_only(w_text) if w_text.strip() else None
                if w_text.strip() and w_g is None:
                    st.warning("Weight must be a number in grams only (e.g., `3245`). No units.")

                notes = st.text_input("Notes (optional)", placeholder="Scale, time relative to feed, etc.")
                submitted = st.form_submit_button("Add weight")
                if submitted:
                    if w_g is None:
                        st.error("Please enter a number in grams (e.g., `3245`). No units.")
                    else:
                        tare_to_store = None
                        if condition == "clean diaper":
                            try:
                                tare_to_store = float(diaper_tare_g) if diaper_tare_g is not None else None
                            except Exception:
                                tare_to_store = None
                            if tare_to_store is not None and tare_to_store > 0:
                                st.session_state["last_diaper_tare_g"] = tare_to_store
                                persist_settings({"last_diaper_tare_g": tare_to_store})

                        e = {
                            "id": str(uuid.uuid4()),
                            "type": "weight",
                            "ts": ts,
                            "weight_g": float(w_g),
                            "condition": condition,
                            "diaper_tare_g": tare_to_store,
                            "notes": notes.strip() if notes else "",
                        }
                        add_event(events, e)
                        flash_and_rerun("Added weight.")

    st.divider()
    st.subheader("Recent events")

    if not events:
        st.info("No events yet.")
        return

    recent_events = sorted(events, key=lambda x: x.get("ts", ""), reverse=True)[:50]
    current_day: Optional[date] = None

    for e in recent_events:
        et = e.get("type", "event")
        tstamp = e.get("ts", "")

        dt: Optional[datetime] = None
        if tstamp:
            try:
                dt = iso_to_dt(tstamp)
            except Exception:
                dt = None

        day_key = dt.date() if dt else None
        if day_key != current_day:
            if current_day is not None:
                st.divider()

            if day_key is None:
                st.markdown("### Unknown date")
            else:
                st.markdown(f"### {dt.strftime('%a')}, {dt.strftime('%b')} {dt.day}, {dt.year}")

            current_day = day_key

        time_str = dt.strftime("%H:%M") if dt is not None else tstamp
        title = f"{time_str} — {pretty_event_label(e)}"

        with st.expander(title, expanded=False):
            ed = iso_to_dt(e["ts"])
            c1, c2 = st.columns(2)
            with c1:
                new_date = st.date_input("Date", value=ed.date(), key=f"ed_d_{e['id']}")
            with c2:
                new_time = time_picker("Time", default=ed.time(), key=f"ed_t_{e['id']}", show_quick=True)

            patch: Dict[str, Any] = {"ts": dt_to_iso(compose_dt(new_date, new_time))}

            if et == "diaper":
                desc = st.multiselect(
                    "Descriptor",
                    options=["pee", "potty", "robust", "light"],
                    default=e.get("descriptor", []),
                    key=f"ed_desc_{e['id']}",
                )
                notes = st.text_input("Notes", value=e.get("notes", ""), key=f"ed_notes_{e['id']}")
                patch.update({"descriptor": desc, "notes": notes})

            elif et == "feeding":
                mode = e.get("feed_mode", "breast")
                mode_ui = st.radio(
                    "Feeding type",
                    ["breast", "bottle"],
                    index=0 if mode == "breast" else 1,
                    horizontal=True,
                    key=f"ed_mode_{e['id']}",
                )
                patch["feed_mode"] = mode_ui

                notes = st.text_input("Notes", value=e.get("notes", ""), key=f"ed_notes_{e['id']}")
                patch["notes"] = notes

                if mode_ui == "bottle":
                    patch["pre_diaper_tare_g"] = None
                    existing_ml = float(e.get("bottle_ml") or 0.0)

                    edit_unit = st.radio(
                        "Amount unit",
                        ["oz", "mL / g (≈ same)"],
                        index=0,
                        horizontal=True,
                        key=f"ed_bottle_unit_{e['id']}",
                    )
                    st.caption("mL and g are treated as equivalent here (1 mL ≈ 1 g).")

                    if edit_unit == "oz":
                        default_amt = float(ml_to_floz(existing_ml)) if existing_ml > 0 else None
                        amt = st.number_input(
                            "Amount (oz)",
                            min_value=0.0,
                            step=0.5,
                            value=default_amt,
                            placeholder="Enter amount",
                            key=f"ed_amt_{e['id']}",
                        )
                        amt_ml = floz_to_ml(float(amt)) if (amt is not None and amt > 0) else 0.0
                        if amt_ml > 0:
                            st.caption(f"≈ {amt_ml:,.0f} mL (≈ g)")
                    else:
                        amt = st.number_input(
                            "Amount (mL or g)",
                            min_value=0.0,
                            step=5.0,
                            value=(existing_ml if existing_ml > 0 else None),
                            placeholder="Enter amount",
                            key=f"ed_amt_{e['id']}",
                        )
                        amt_ml = float(amt) if (amt is not None and amt > 0) else 0.0
                        if amt_ml > 0:
                            st.caption(f"≈ {ml_to_floz(amt_ml):.2f} oz")

                    patch.update({"bottle_ml": float(amt_ml) if amt_ml > 0 else None, "delta_g": None})
                    patch.setdefault("pre_weight_g", e.get("pre_weight_g"))
                    patch.setdefault("pre_condition", e.get("pre_condition"))
                    patch.setdefault("post_weight_g", e.get("post_weight_g"))

                else:
                    cur_side = (e.get("start_side") or "").upper().strip()
                    if cur_side not in {"L", "R"}:
                        _, suggested_next = breast_side_status(events)
                        cur_side = suggested_next

                    start_side = st.radio(
                        "Starting side (required)",
                        ["L", "R"],
                        index=0 if cur_side == "L" else 1,
                        horizontal=True,
                        key=f"ed_startside_{e['id']}",
                    )

                    new_delta_txt = st.text_input(
                        "Delta (g)",
                        value=fmt_g(e.get("delta_g")),
                        key=f"ed_delta_{e['id']}",
                    )
                    new_delta_g = None
                    if new_delta_txt.strip():
                        new_delta_g = parse_number_only(new_delta_txt)
                        if new_delta_g is None:
                            st.warning("Delta must be a number in grams only (e.g., `32` or `32.5`). No units.")
                        else:
                            st.caption(f"≈ {ml_to_floz(new_delta_g):.2f} oz (assuming 1 g ≈ 1 mL)")

                    patch.update(
                        {
                            "bottle_ml": None,
                            "start_side": start_side,
                            "delta_g": float(new_delta_g) if new_delta_g is not None else None,
                        }
                    )


            elif et == "weight":
                w_txt = st.text_input("Weight (g)", value=fmt_g(e.get("weight_g")), key=f"ed_w_{e['id']}")
                cond = st.radio(
                    "Condition",
                    ["naked", "clean diaper"],
                    index=0 if e.get("condition") == "naked" else 1,
                    horizontal=True,
                    key=f"ed_wcond_{e['id']}",
                )

                if cond == "clean diaper":
                    default_tare = e.get("diaper_tare_g")
                    if default_tare is None:
                        default_tare = st.session_state.get("last_diaper_tare_g", DIAPER_TARE_G)
                    tare = st.number_input(
                        "Diaper tare (g) for this event",
                        min_value=0.0,
                        step=1.0,
                        value=float(default_tare),
                        key=f"ed_wtare_{e['id']}",
                    )
                    patch["diaper_tare_g"] = float(tare) if tare and tare > 0 else None
                else:
                    patch["diaper_tare_g"] = None

                notes = st.text_input("Notes", value=e.get("notes", ""), key=f"ed_notes_{e['id']}")
                w_g = parse_number_only(w_txt) if w_txt.strip() else None
                if w_txt.strip() and w_g is None:
                    st.warning("Weight must be a number in grams only (e.g., `3245`). No units.")
                patch.update({"weight_g": float(w_g) if w_g is not None else None, "condition": cond, "notes": notes})

            elif et == "pumping":
                existing_total_ml = float(e.get("pump_total_ml") or 0.0)
                existing_left_ml = float(e.get("pump_left_ml") or 0.0)
                existing_right_ml = float(e.get("pump_right_ml") or 0.0)

                edit_unit = st.radio(
                    "Amount unit",
                    ["oz", "mL / g (≈ same)"],
                    index=0,
                    horizontal=True,
                    key=f"ed_pump_unit_{e['id']}",
                )
                st.caption("mL and g are treated as equivalent here (1 mL ≈ 1 g).")

                if edit_unit == "oz":
                    def_ml_to_oz = lambda ml: float(ml_to_floz(ml)) if ml and ml > 0 else None  # noqa: E731

                    total = st.number_input(
                        "Total (oz)",
                        min_value=0.0,
                        step=0.5,
                        value=def_ml_to_oz(existing_total_ml),
                        placeholder="e.g. 4",
                        key=f"ed_pump_total_{e['id']}",
                    )
                    left = st.number_input(
                        "Left (oz)",
                        min_value=0.0,
                        step=0.5,
                        value=def_ml_to_oz(existing_left_ml),
                        placeholder="e.g. 2",
                        key=f"ed_pump_left_{e['id']}",
                    )
                    right = st.number_input(
                        "Right (oz)",
                        min_value=0.0,
                        step=0.5,
                        value=def_ml_to_oz(existing_right_ml),
                        placeholder="e.g. 2",
                        key=f"ed_pump_right_{e['id']}",
                    )

                    total_ml = floz_to_ml(float(total)) if (total is not None and total > 0) else None
                    left_ml = floz_to_ml(float(left)) if (left is not None and left > 0) else None
                    right_ml = floz_to_ml(float(right)) if (right is not None and right > 0) else None

                else:
                    total = st.number_input(
                        "Total (mL or g)",
                        min_value=0.0,
                        step=5.0,
                        value=(existing_total_ml if existing_total_ml > 0 else None),
                        placeholder="e.g. 120",
                        key=f"ed_pump_total_{e['id']}",
                    )
                    left = st.number_input(
                        "Left (mL or g)",
                        min_value=0.0,
                        step=5.0,
                        value=(existing_left_ml if existing_left_ml > 0 else None),
                        placeholder="e.g. 60",
                        key=f"ed_pump_left_{e['id']}",
                    )
                    right = st.number_input(
                        "Right (mL or g)",
                        min_value=0.0,
                        step=5.0,
                        value=(existing_right_ml if existing_right_ml > 0 else None),
                        placeholder="e.g. 60",
                        key=f"ed_pump_right_{e['id']}",
                    )

                    total_ml = float(total) if (total is not None and total > 0) else None
                    left_ml = float(left) if (left is not None and left > 0) else None
                    right_ml = float(right) if (right is not None and right > 0) else None

                if total_ml is None and (left_ml is not None or right_ml is not None):
                    total_ml = (left_ml or 0.0) + (right_ml or 0.0)
                    if total_ml <= 0:
                        total_ml = None

                patch.update(
                    {
                        "pump_total_ml": float(total_ml) if (total_ml is not None and total_ml > 0) else None,
                        "pump_left_ml": float(left_ml) if (left_ml is not None and left_ml > 0) else None,
                        "pump_right_ml": float(right_ml) if (right_ml is not None and right_ml > 0) else None,
                        "notes": e.get("notes", ""),
                    }
                )

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Save changes", key=f"save_{e['id']}"):
                    update_event(events, e["id"], patch)

                    if patch.get("pre_diaper_tare_g") is not None:
                        st.session_state["last_diaper_tare_g"] = float(patch["pre_diaper_tare_g"])
                        persist_settings({"last_diaper_tare_g": float(patch["pre_diaper_tare_g"])})

                    if patch.get("diaper_tare_g") is not None:
                        st.session_state["last_diaper_tare_g"] = float(patch["diaper_tare_g"])
                        persist_settings({"last_diaper_tare_g": float(patch["diaper_tare_g"])})

                    st.success("Saved.")
                    st.rerun()

            with b2:
                if st.button("Delete", key=f"del_{e['id']}"):
                    delete_event(events, e["id"])
                    st.warning("Deleted.")
                    st.rerun()

    st.download_button(
        "Download events.json",
        data=json.dumps(load_events(), indent=2).encode("utf-8"),
        file_name="events.json",
        mime="application/json",
    )


def render_graph_tab() -> None:
    events = load_events()

    st.caption(
        "Plots only **naked** and **diaper-only** weights. "
        f"Diaper-only weights subtract the event’s recorded diaper tare (default {DIAPER_TARE_G:.0f} g)."
    )

    if not events:
        st.info("No events to graph yet.")
        return

    # -------------------------------------------------------------------------
    # Weight over time (event-based axis)
    # -------------------------------------------------------------------------
    rows = []
    for e in sorted(events, key=lambda x: x.get("ts", "")):
        ts = e.get("ts")
        dt = iso_to_dt(ts) if ts else None
        rows.append(
            {
                "event_idx": len(rows) + 1,
                "time": dt,
                "type": e.get("type"),
                "weight_g": weight_for_event_row(e),
                "notes": e.get("notes", ""),
            }
        )

    df = pd.DataFrame(rows)

    st.subheader("Weight over time")

    df_w = df.dropna(subset=["time", "weight_g"]).copy()
    if df_w.empty:
        st.info("No plot-worthy weights yet (only naked / clean-diaper weights are plotted).")
    else:
        df_w = df_w.sort_values("time")

        y_units = st.radio("Y-axis units", ["g", "lb/oz"], horizontal=True, key="weight_y_units")

        df_w["weight_lb"] = df_w["weight_g"] / LB_TO_G
        df_w["weight_lb_oz"] = df_w["weight_g"].apply(
            lambda g: (lambda lo: f"{lo[0]} lb {lo[1]:.1f} oz")(grams_to_lbs_oz(float(g)))
        )

        if y_units == "g":
            v_min = float(df_w["weight_g"].min())
            v_max = float(df_w["weight_g"].max())
            span = max(v_max - v_min, 1.0)
            pad = max(25.0, span * 0.05)
            y_domain = [v_min - pad, v_max + pad]

            y_enc = alt.Y(
                "weight_g:Q",
                title="Weight (g)",
                scale=alt.Scale(domain=y_domain, nice=True, zero=False),
            )
        else:
            v_min = float(df_w["weight_lb"].min())
            v_max = float(df_w["weight_lb"].max())
            span = max(v_max - v_min, 1e-6)
            pad = max(0.05, span * 0.05)
            y_domain = [v_min - pad, v_max + pad]

            y_enc = alt.Y(
                "weight_lb:Q",
                title="Weight (lb/oz)",
                scale=alt.Scale(domain=y_domain, nice=True, zero=False),
                axis=alt.Axis(
                    labelExpr="floor(datum.value) + ' lb ' + format((datum.value - floor(datum.value))*16, '.1f') + ' oz'"
                ),
            )

        x_enc = alt.X(
            "time:T",
            title="Date",
            axis=alt.Axis(format="%m/%d", labelAngle=0),
        )

        line = (
            alt.Chart(df_w)
            .mark_line()
            .encode(
                x=x_enc,
                y=y_enc,
                tooltip=[
                    alt.Tooltip("time:T", title="Time"),
                    alt.Tooltip("type:N", title="Event type"),
                    alt.Tooltip("weight_g:Q", title="Weight (g)", format=".0f"),
                    alt.Tooltip("weight_lb_oz:N", title="Weight (lb/oz)"),
                    alt.Tooltip("notes:N", title="Notes"),
                ],
            )
        )

        pts = (
            alt.Chart(df_w)
            .mark_point()
            .encode(
                x=x_enc,
                y=y_enc,
                tooltip=[
                    alt.Tooltip("time:T", title="Time"),
                    alt.Tooltip("weight_g:Q", title="Weight (g)", format=".0f"),
                    alt.Tooltip("weight_lb_oz:N", title="Weight (lb/oz)"),
                ],
            )
        )

        st.altair_chart((line + pts).properties(height=320), width="stretch")

    st.divider()

    # -------------------------------------------------------------------------
    # Daily intake (last 3 days overlaid)
    # -------------------------------------------------------------------------
    st.subheader("Daily intake (last 3 days overlaid)")

    today = date.today()
    days = [today - timedelta(days=2), today - timedelta(days=1), today]
    days_set = set(days)

    feed_rows = []
    for e in sorted(events, key=lambda x: x.get("ts", "")):
        ts = e.get("ts")
        if not ts:
            continue
        dt = iso_to_dt(ts)
        if dt.date() not in days_set:
            continue
        ml = feeding_intake_ml(e)
        if ml is None:
            continue

        mode = str(e.get("feed_mode") or "").strip().lower()
        feed_rows.append({"time": dt, "intake_ml": float(ml), "mode": mode})


    if not feed_rows:
        st.info("No feeding amounts recorded in the last 3 days.")
        return

    df_feed = pd.DataFrame(feed_rows).sort_values("time")
    df_feed["day"] = df_feed["time"].dt.date
    df_feed["minute"] = df_feed["time"].dt.hour * 60 + df_feed["time"].dt.minute + df_feed["time"].dt.second / 60.0

    df_feed = df_feed.sort_values(["day", "minute"])
    df_feed["cum_ml"] = df_feed.groupby("day")["intake_ml"].cumsum()

    midnights = pd.DataFrame([{"day": d, "minute": 0.0, "cum_ml": 0.0} for d in days])

    now_dt = datetime.now()
    now_min = now_dt.hour * 60 + now_dt.minute + now_dt.second / 60.0
    now_min = float(min(max(now_min, 0.0), 24 * 60))

    endpoints = []
    for d in days:
        day_data = df_feed[df_feed["day"] == d]
        final_ml = float(day_data["cum_ml"].iloc[-1]) if not day_data.empty else 0.0
        if d == today:
            endpoints.append({"day": d, "minute": now_min, "cum_ml": final_ml})
        else:
            endpoints.append({"day": d, "minute": 24 * 60, "cum_ml": final_ml})

    df_line = pd.concat(
        [df_feed[["day", "minute", "cum_ml"]], midnights, pd.DataFrame(endpoints)],
        ignore_index=True,
    ).sort_values(["day", "minute"])

    def _day_label(d0: date) -> str:
        if d0 == today:
            prefix = "Today"
        elif d0 == (today - timedelta(days=1)):
            prefix = "Yesterday"
        else:
            prefix = "2 days ago"
        return f"{prefix} ({d0.strftime('%a')} {d0.strftime('%b')} {d0.day})"

    label_map = {d: _day_label(d) for d in days}
    df_line["Day"] = df_line["day"].map(label_map)
    df_pts = df_feed.copy()
    df_pts["Day"] = df_pts["day"].map(label_map)

    units = st.radio("Units", ["mL", "oz"], horizontal=True, key="intake_units")

    goal_floz_today = None
    if st.session_state.get("goal_date") == str(today):
        g = st.session_state.get("goal_floz")
        goal_floz_today = float(g) if g is not None else None

    today_total_ml = float(df_feed[df_feed["day"] == today]["intake_ml"].sum()) if (df_feed["day"] == today).any() else 0.0
    today_total_floz = ml_to_floz(today_total_ml)

    now_dt2 = datetime.now()
    midnight2 = datetime.combine(today, time(0, 0, 0))
    elapsed_hours = max((now_dt2 - midnight2).total_seconds() / 3600.0, 1e-6)



    if units == "oz":
        df_line["Value"] = df_line["cum_ml"].apply(ml_to_floz)
        df_pts["Value"] = df_pts["cum_ml"].apply(ml_to_floz)
        y_title = "Cumulative intake (oz)"
        val_fmt = ".2f"
    else:
        df_line["Value"] = df_line["cum_ml"]
        df_pts["Value"] = df_pts["cum_ml"]
        y_title = "Cumulative intake (mL)"
        val_fmt = ".0f"

    hour_ticks = list(range(0, 24 * 60 + 1, 60))  # 0, 60, 120, ... 1440

    x_enc = alt.X(
        "minute:Q",
        title="Time of day",
        scale=alt.Scale(domain=[0, 24 * 60], nice=False, zero=True),
        axis=alt.Axis(
            values=hour_ticks,  # force ticks at exact hours only
            labelExpr="format(floor(datum.value/60), '02d') + ':00'",
            labelOverlap="greedy",
        ),
    )

    line = (
        alt.Chart(df_line)
        .mark_line()
        .encode(
            x=x_enc,
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color(
                "Day:N",
                title=None,
                sort=alt.SortField(field="day", order="ascending"),
            ),
            tooltip=[
                alt.Tooltip("Day:N", title="Day"),
                alt.Tooltip("minute:Q", title="Time (min)"),
                alt.Tooltip("Value:Q", title="Cumulative", format=val_fmt),
            ],
        )
    )

    points = (
        alt.Chart(df_pts)
        .mark_point()
        .encode(
            x="minute:Q",
            y="Value:Q",
            color=alt.Color(
                "Day:N",
                title=None,
                sort=alt.SortField(field="day", order="ascending"),
            ),
            shape=alt.Shape(
                "mode:N",
                title=None,
                scale=alt.Scale(domain=["breast", "bottle"], range=["circle", "square"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Day:N", title="Day"),
                alt.Tooltip("time:T", title="Time"),
                alt.Tooltip("mode:N", title="Type"),
                alt.Tooltip("Value:Q", title="Cumulative", format=val_fmt),
            ],
        )
    )


    goal_layer = None
    if goal_floz_today is not None and goal_floz_today > 0:
        goal_val = goal_floz_today if units == "oz" else floz_to_ml(goal_floz_today)
        goal_layer = (
            alt.Chart(pd.DataFrame({"goal": [goal_val]}))
            .mark_rule(strokeDash=[6, 4])
            .encode(
                y=alt.datum(goal_val),
                tooltip=[alt.Tooltip("goal:Q", title="Goal", format=val_fmt)],
            )
        )

    chart = line + points
    if goal_layer is not None:
        chart = (chart + goal_layer).resolve_scale(y="shared")

    st.altair_chart(chart.properties(height=320), width="stretch")


    if goal_floz_today is not None and goal_floz_today > 0:
        remaining_floz = max(goal_floz_today - today_total_floz, 0.0)
        remaining_hours = max(24.0 - elapsed_hours, 1e-6)

        pct = min(today_total_floz / goal_floz_today, 1.0)

        actual_oz_per_hr = today_total_floz / elapsed_hours
        goal_oz_per_hr = goal_floz_today / 24.0
        needed_oz_per_hr = remaining_floz / remaining_hours

        expected_by_now = goal_floz_today * (elapsed_hours / 24.0)
        on_track_delta = today_total_floz - expected_by_now

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.metric("Today so far", f"{today_total_floz:.2f} oz")
        with r1c2:
            st.metric("Goal", f"{goal_floz_today:.1f} oz")
        with r1c3:
            st.metric("Remaining", f"{remaining_floz:.2f} oz")

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.metric("Actual pace", f"{actual_oz_per_hr:.2f} oz/hr")
        with r2c2:
            st.metric("Goal pace", f"{goal_oz_per_hr:.2f} oz/hr")
        with r2c3:
            st.metric("Needed pace", f"{needed_oz_per_hr:.2f} oz/hr")

        st.progress(pct)

        if on_track_delta >= 0:
            st.caption(f"On track (ahead by {on_track_delta:.2f} oz vs pace-to-goal by now).")
        else:
            st.caption(f"Behind pace (need {abs(on_track_delta):.2f} oz to be on pace by now).")
        
        # Yesterday-by-now (actual logged feeds) + next feed time yesterday
        yday = today - timedelta(days=1)
        yday_df = df_feed[df_feed["day"] == yday].sort_values("minute")

        if yday_df.empty:
            st.caption("Yesterday by this time: no feeds recorded.")
        else:
            upto = yday_df[yday_df["minute"] <= now_min]
            yday_by_now_ml = float(upto["cum_ml"].iloc[-1]) if not upto.empty else 0.0

            next_row = yday_df[yday_df["minute"] > now_min].head(1)
            next_time_str = (
                next_row["time"].iloc[0].strftime("%H:%M") if not next_row.empty else "no later feed recorded"
            )

            if units == "oz":
                amt_str = f"{ml_to_floz(yday_by_now_ml):.2f} oz"
            else:
                amt_str = f"{yday_by_now_ml:,.0f} mL"

            st.caption(f"Yesterday by this time: {amt_str}; next feed was at {next_time_str}.")

    else:
        st.caption("Set a feeding goal in the Calculator tab to show a goal line + progress here.")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Today so far", f"{today_total_floz:.2f} oz")
        with m2:
            st.metric("Actual pace", f"{(today_total_floz / elapsed_hours):.2f} oz/hr")
        with m3:
            st.metric("Today so far", f"{today_total_ml:,.0f} mL")


    daily = df_feed.groupby("day", as_index=False)["intake_ml"].sum()
    daily["intake_oz"] = daily["intake_ml"].apply(ml_to_floz)
    daily = daily.sort_values("day")
    st.dataframe(daily[["day", "intake_oz"]], hide_index=True, width="stretch")





    with st.expander("Info"):
        st.write(
            "- Only **random weight** events are plotted.\n"
            f"- Diaper-only weights subtract the event’s recorded diaper tare (default **{DIAPER_TARE_G:.0f} g**)."
        )



def render_thank_you_tab() -> None:
    st.subheader("Thank you notes")
    st.caption("Track whether you’ve sent a thank-you yet. Stored locally in `data/thank_you.json`.")

    items = load_thank_yous()

    # Migrate older schema -> compact schema
    migrated = False
    out: List[Dict[str, Any]] = []
    for it in items:
        rid = it.get("id") or str(uuid.uuid4())
        created_ts = it.get("created_ts") or dt_to_iso(datetime.now())
        person = (it.get("person") or "").strip()
        gift = (it.get("gift") or "").strip()

        if "thanked" in it:
            thanked = bool(it.get("thanked"))
        else:
            resp = it.get("response")
            thanked = True if resp in {"text", "thank you note"} else False
            migrated = True

        if not person:
            migrated = True
            continue

        out.append({"id": rid, "created_ts": created_ts, "person": person, "gift": gift, "thanked": thanked})

    items = out
    items = sorted(items, key=lambda x: (bool(x.get("thanked")), x.get("created_ts", "")))

    if migrated:
        save_thank_yous(items)

    with st.container(border=True):
        st.markdown("#### Add entry")
        with st.form("add_thank_you_compact", clear_on_submit=True):
            c1, c2 = st.columns([1.3, 2.2])
            with c1:
                person = st.text_input("Person", placeholder="e.g. Bumi")
            with c2:
                gift = st.text_input("Gift (optional)", placeholder="e.g. Baby blanket")

            submitted = st.form_submit_button("Add")
            if submitted:
                if not person.strip():
                    st.error("Person is required.")
                else:
                    items.append(
                        {
                            "id": str(uuid.uuid4()),
                            "created_ts": dt_to_iso(datetime.now()),
                            "person": person.strip(),
                            "gift": gift.strip() if gift else "",
                            "thanked": False,
                        }
                    )
                    save_thank_yous(items)
                    flash_and_rerun("Added thank-you entry.")

    st.divider()

    if not items:
        st.info("No thank-you entries yet.")
        return

    df = pd.DataFrame([{"Person": it["person"], "Gift": it.get("gift", ""), "Thanked": bool(it.get("thanked"))} for it in items])

    edited = st.data_editor(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Person": st.column_config.TextColumn("Person", required=True),
            "Gift": st.column_config.TextColumn("Gift"),
            "Thanked": st.column_config.CheckboxColumn("Thanked"),
        },
        disabled=[],
    )

    b1, b2 = st.columns([1.2, 2.2])

    with b1:
        if st.button("Save", key="ty_save_compact"):
            new_items: List[Dict[str, Any]] = []
            for i, row in edited.iterrows():
                person = str(row.get("Person") or "").strip()
                if not person:
                    continue
                new_items.append(
                    {
                        "id": items[i]["id"],
                        "created_ts": items[i].get("created_ts") or dt_to_iso(datetime.now()),
                        "person": person,
                        "gift": str(row.get("Gift") or "").strip(),
                        "thanked": bool(row.get("Thanked")),
                    }
                )
            new_items = sorted(new_items, key=lambda x: (bool(x.get("thanked")), x.get("created_ts", "")))
            save_thank_yous(new_items)
            flash_and_rerun("Saved thank-you list.")

    with b2:
        labels = [f"{it['person']}" + (f" — {it.get('gift','')}" if it.get("gift") else "") for it in items]
        choice = st.selectbox("Delete", options=["(none)"] + labels, index=0, key="ty_del_pick")
        if st.button("Delete selected", key="ty_del_btn", disabled=(choice == "(none)")):
            idx = labels.index(choice)
            new_items = [it for j, it in enumerate(items) if j != idx]
            save_thank_yous(new_items)
            flash_and_rerun("Deleted entry.", icon="🗑️")


def render_advice_tab() -> None:
    st.subheader("Notes & advice")
    st.caption("Save the advice you want to remember. Stored locally in `data/advice.json`.")

    items = load_advice()

    # Normalize schema (best-effort)
    migrated = False
    norm: List[Dict[str, Any]] = []
    status = "Untried"  # preserved behavior: name exists for later use

    for it in items:
        rid = it.get("id") or str(uuid.uuid4())
        created_ts = it.get("created_ts") or dt_to_iso(datetime.now())

        text = (it.get("text") or it.get("advice") or "").strip()
        person = (it.get("person") or it.get("from") or "").strip()
        category = (it.get("category") or "General").strip() or "General"
        if category not in ADVICE_CATEGORIES:
            category = "General"

        notes = (it.get("notes") or "").strip()

        status = (it.get("status") or "Untried").strip()
        if status not in ADVICE_STATUSES:
            status = "Untried"

        if not text:
            migrated = True
            continue

        norm.append(
            {
                "id": rid,
                "created_ts": created_ts,
                "text": text,
                "person": person,
                "category": category,
                "notes": notes,
                "status": status,
            }
        )

        if any(k in it for k in ("advice", "from", "tags", "keep", "pinned", "tried")):
            migrated = True

    items = norm
    if migrated:
        save_advice(items)

    # Quick add (single-line)
    st.session_state.setdefault("advice_quick_add", "")
    _cat_map = {c.lower(): c for c in ADVICE_CATEGORIES}

    def _quick_add_advice() -> None:
        raw = (st.session_state.get("advice_quick_add") or "").strip()
        if not raw:
            return

        category0 = "General"
        text0 = raw
        if ":" in raw:
            head, tail = raw.split(":", 1)
            head_norm = head.strip().lower()
            if head_norm in _cat_map and tail.strip():
                category0 = _cat_map[head_norm]
                text0 = tail.strip()

        new_item = {
            "id": str(uuid.uuid4()),
            "created_ts": dt_to_iso(datetime.now()),
            "text": text0,
            "person": "",
            "category": category0,
            "notes": "",
            "status": "Untried",
        }

        cur = load_advice()
        cur.append(new_item)
        save_advice(cur)

        st.session_state["advice_quick_add"] = ""
        flash("Saved advice.")

    with st.container(border=True):
        st.markdown("#### Quick add")
        st.text_input(
            "Type advice and press Enter",
            key="advice_quick_add",
            placeholder="e.g. Sleep: Try white noise before troubleshooting anything else",
            help="Optional: start with 'Sleep:', 'Feeding:', etc. to auto-categorize.",
            on_change=_quick_add_advice,
        )

    st.divider()

    # Rich add
    with st.container(border=True):
        st.markdown("#### Add entry")
        with st.form("advice_add_form", clear_on_submit=True):
            c1, c2 = st.columns([2.2, 1.2])
            with c1:
                text = st.text_area(
                    "Advice",
                    placeholder="e.g. “If you think they’re hungry, offer a feed before troubleshooting anything else.”",
                    height=90,
                )
            with c2:
                person = st.text_input("From (optional)", placeholder="e.g. Lactation consultant")
                category = st.selectbox("Category", ADVICE_CATEGORIES, index=0)

            notes = st.text_input("Notes (optional)", placeholder="Context / what happened / whether it helped")
            submitted = st.form_submit_button("Add")
            if submitted:
                if not text.strip():
                    st.error("Advice text is required.")
                else:
                    new_item = {
                        "id": str(uuid.uuid4()),
                        "created_ts": dt_to_iso(datetime.now()),
                        "text": text.strip(),
                        "person": person.strip(),
                        "category": category,
                        "notes": notes.strip(),
                        "status": status,
                    }
                    items2 = load_advice()
                    items2.append(new_item)
                    save_advice(items2)
                    flash_and_rerun("Saved advice.")

    st.divider()

    # Search + filters
    with st.container(border=True):
        st.markdown("#### Find")
        f1, f2 = st.columns([2.4, 1.4])
        with f1:
            q = st.text_input("Search", placeholder="filter by text / from / notes")
        with f2:
            cat = st.selectbox("Category filter", ["All"] + ADVICE_CATEGORIES, index=0)
            status_filter = st.selectbox("Status filter", ["All"] + ADVICE_STATUSES, index=0)

    def _matches(it: Dict[str, Any]) -> bool:
        if cat != "All" and it.get("category") != cat:
            return False
        if status_filter != "All" and (it.get("status") or "Untried") != status_filter:
            return False
        if q.strip():
            needle = q.strip().lower()
            hay = " ".join(
                [
                    str(it.get("text") or ""),
                    str(it.get("person") or ""),
                    str(it.get("category") or ""),
                    str(it.get("notes") or ""),
                ]
            ).lower()
            return needle in hay
        return True

    view = [it for it in load_advice() if _matches(it)]
    view = sorted(view, key=lambda x: x.get("created_ts", ""), reverse=True)

    if not view:
        st.info("No advice saved yet." if not load_advice() else "No matches for your filters.")
        return

    st.markdown("#### Saved advice")

    for it in view[:200]:
        person = it.get("person") or "—"
        category = it.get("category") or "General"

        preview = (it.get("text") or "").strip().replace("\n", " ")
        if len(preview) > 70:
            preview = preview[:70].rstrip() + "…"

        st_status = it.get("status") or "Untried"
        title = f"{category} — {st_status} — {person} — {preview}"

        with st.expander(title, expanded=False):
            c1, c2 = st.columns([1.6, 1.2])
            with c1:
                new_person = st.text_input("From", value=str(it.get("person") or ""), key=f"adv_p_{it['id']}")
            with c2:
                new_cat = st.selectbox(
                    "Category",
                    ADVICE_CATEGORIES,
                    index=ADVICE_CATEGORIES.index(it.get("category")) if it.get("category") in ADVICE_CATEGORIES else 0,
                    key=f"adv_c_{it['id']}",
                )

                cur_status = it.get("status") or "Untried"
                if cur_status not in ADVICE_STATUSES:
                    cur_status = "Untried"

                new_status = st.selectbox(
                    "Status",
                    ADVICE_STATUSES,
                    index=ADVICE_STATUSES.index(cur_status),
                    key=f"adv_s_{it['id']}",
                )

            new_text = st.text_area("Advice", value=str(it.get("text") or ""), height=120, key=f"adv_txt_{it['id']}")
            new_notes = st.text_input(
                "Notes (optional)",
                value=str(it.get("notes") or ""),
                placeholder="Context / outcome",
                key=f"adv_notes_{it['id']}",
            )

            try:
                created_dt = iso_to_dt(it.get("created_ts"))
                st.caption(f"Saved: {created_dt.strftime('%b')} {created_dt.day}, {created_dt.strftime('%H:%M')}")
            except Exception:
                pass

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Save changes", key=f"adv_save_{it['id']}"):
                    patch = {
                        "person": new_person.strip(),
                        "category": new_cat,
                        "text": new_text.strip(),
                        "notes": new_notes.strip(),
                        "status": new_status,
                    }
                    if not patch["text"]:
                        st.error("Advice text can’t be blank.")
                    else:
                        items_all = load_advice()
                        update_advice(items_all, it["id"], patch)
                        flash_and_rerun("Saved.")
            with b2:
                if st.button("Delete", key=f"adv_del_{it['id']}"):
                    items_all = load_advice()
                    delete_advice(items_all, it["id"])
                    flash_and_rerun("Deleted.", icon="🗑️")


def render_groceries_tab() -> None:
    st.subheader("Groceries")
    st.caption("Simple per-store lists with add + cross-off + delete. Stored locally in `data/groceries.json`.")

    def _slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    def _set_done(item_id: str) -> None:
        k = f"gro_done__{item_id}"
        new_val = bool(st.session_state.get(k, False))
        cur = load_groceries()
        for it in cur:
            if it.get("id") == item_id:
                it["done"] = new_val
                break
        save_groceries(cur)

    def _delete_item(item_id: str) -> None:
        cur = [x for x in load_groceries() if x.get("id") != item_id]
        save_groceries(cur)

    def _toggle_priority_btn(item_id: str) -> None:
        cur = load_groceries()
        for it in cur:
            if it.get("id") == item_id:
                it["priority"] = not bool(it.get("priority", False))
                break
        save_groceries(cur)
        flash_and_rerun("Updated priority.", icon="⭐")

    for store in GROCERY_STORES:
        store_key = _slug(store)

        with st.container(border=True):
            st.markdown(f"### {store}")

            with st.form(f"gro_add__{store_key}", clear_on_submit=True):
                text = st.text_input(
                    "Add item",
                    placeholder="e.g. bananas",
                    key=f"gro_text__{store_key}",
                    label_visibility="collapsed",
                )

                submitted = st.form_submit_button("Add")
                if submitted:
                    if not text.strip():
                        st.error("Item is required.")
                    else:
                        cur = load_groceries()
                        cur.append(
                            {
                                "id": str(uuid.uuid4()),
                                "created_ts": dt_to_iso(datetime.now()),
                                "text": text.strip(),
                                "store": store,
                                "done": False,
                                "priority": False,
                            }
                        )
                        save_groceries(cur)
                        flash_and_rerun(f"Added to {store}.")

            items = load_groceries()
            store_items = [it for it in items if (it.get("store") or "Grocery/Any") == store]

            store_items = sorted(
                store_items,
                key=lambda x: (bool(x.get("done")), not bool(x.get("priority", False)), x.get("created_ts", "")),
            )

            if not store_items:
                st.caption("No items.")
                continue

            for it in store_items:
                iid = str(it.get("id"))
                done = bool(it.get("done"))
                pri = bool(it.get("priority", False))
                text = str(it.get("text") or "").strip()

                c_done, c_text, c_actions = st.columns([0.35, 3.0, 0.8])

                with c_done:
                    st.checkbox(
                        "done",
                        value=done,
                        key=f"gro_done__{iid}",
                        on_change=_set_done,
                        args=(iid,),
                        label_visibility="collapsed",
                    )

                with c_text:
                    st.markdown(f"~~{text}~~" if done else text)

                with c_actions:
                    a_star, a_del = st.columns([1, 1])

                    with a_star:
                        star = "⭐" if pri else "☆"
                        if st.button(star, key=f"gro_pri_btn__{iid}", help="Toggle priority (shows on Snapshot)."):
                            _toggle_priority_btn(iid)

                    with a_del:
                        if st.button("🗑️", key=f"gro_del__{iid}"):
                            _delete_item(iid)
                            flash_and_rerun("Deleted item.", icon="🗑️")


# =============================================================================
# App entrypoint
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="Baby Weight + Daily Intake", page_icon="🍼", layout="centered")

    # Hydrate persisted settings into session_state
    s = _settings_cache()
    st.session_state.setdefault("calc_weight_text", s.get("calc_weight_text", ""))
    st.session_state.setdefault("goal_floz", s.get("goal_floz", None))
    st.session_state.setdefault("goal_date", s.get("goal_date", str(date.today())))
    st.session_state.setdefault("calc_weight_source", s.get("calc_weight_source", "Recent"))
    st.session_state.setdefault("calc_recent_idx", s.get("calc_recent_idx", 0))
    st.session_state.setdefault("weight_y_units", "lb/oz")
    st.session_state.setdefault("intake_units", "oz")


    # Last-used diaper tare (default fallback)
    _last_tare = s.get("last_diaper_tare_g", DIAPER_TARE_G)
    try:
        _last_tare = float(_last_tare)
    except Exception:
        _last_tare = float(DIAPER_TARE_G)
    if _last_tare <= 0:
        _last_tare = float(DIAPER_TARE_G)

    st.session_state.setdefault("last_diaper_tare_g", _last_tare)

    st.title("🍼 Baby Weight & Daily Intake Calculator")

    # One-shot toast across reruns
    if "_flash" in st.session_state:
        msg, icon = st.session_state.pop("_flash")
        try:
            st.toast(msg, icon=icon)
        except Exception:
            st.success(msg)

    tabs = st.tabs(
        ["Snapshot", "Calculator", "Event Tracker", "Graph", "Thank You Notes", "Notes & Advice", "Groceries"]
    )

    with tabs[0]:
        render_snapshot_tab()

    with tabs[1]:
        render_calculator_tab()

    with tabs[2]:
        render_event_tracker_tab()

    with tabs[3]:
        render_graph_tab()

    with tabs[4]:
        render_thank_you_tab()

    with tabs[5]:
        render_advice_tab()

    with tabs[6]:
        render_groceries_tab()


if __name__ == "__main__":
    main()
