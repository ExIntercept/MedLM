"""Per-user JSON storage. One directory per account, everything human-readable."""
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import USERS_DIR

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")

EMPTY_PROFILE: Dict[str, Any] = {
    "display_name": "",
    "age": "",
    "sex_at_birth": "",
    "height_cm": "",
    "weight_kg": "",
    "conditions": [],
    "medications": [],
    "allergies": [],
    "notes": "",
}


def valid_username(name: str) -> bool:
    return bool(USERNAME_RE.match(name or ""))


def user_dir(username: str) -> Path:
    return USERS_DIR / username


def user_exists(username: str) -> bool:
    return (user_dir(username) / "account.json").exists()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


# --- account -----------------------------------------------------------------

def read_account(username: str) -> Optional[Dict[str, Any]]:
    return _read_json(user_dir(username) / "account.json", None)


def write_account(username: str, account: Dict[str, Any]) -> None:
    _write_json(user_dir(username) / "account.json", account)


def create_user(username: str, account: Dict[str, Any]) -> None:
    write_account(username, account)
    _write_json(user_dir(username) / "profile.json", dict(EMPTY_PROFILE))
    _write_json(user_dir(username) / "memory.json", {"facts": [], "suggested": []})
    _write_json(user_dir(username) / "chat.json", {"messages": []})


# --- profile -----------------------------------------------------------------

def read_profile(username: str) -> Dict[str, Any]:
    profile = _read_json(user_dir(username) / "profile.json", {})
    merged = dict(EMPTY_PROFILE)
    merged.update(profile or {})
    return merged


def write_profile(username: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(EMPTY_PROFILE)
    for key in clean:
        if key in profile:
            clean[key] = profile[key]
    for list_key in ("conditions", "medications", "allergies"):
        value = clean[list_key]
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        clean[list_key] = [str(v).strip() for v in value if str(v).strip()]
    _write_json(user_dir(username) / "profile.json", clean)
    return clean


# --- memory ------------------------------------------------------------------

def read_memory(username: str) -> Dict[str, Any]:
    mem = _read_json(user_dir(username) / "memory.json", {"facts": [], "suggested": []})
    mem.setdefault("facts", [])
    mem.setdefault("suggested", [])
    return mem


def write_memory(username: str, memory: Dict[str, Any]) -> None:
    _write_json(user_dir(username) / "memory.json", memory)


def add_fact(username: str, text: str, source: str = "manual") -> Dict[str, Any]:
    mem = read_memory(username)
    fact = {
        "id": uuid.uuid4().hex[:8],
        "text": text.strip(),
        "source": source,
        "ts": time.time(),
    }
    mem["facts"].append(fact)
    write_memory(username, mem)
    return fact


def delete_fact(username: str, fact_id: str) -> None:
    mem = read_memory(username)
    mem["facts"] = [f for f in mem["facts"] if f["id"] != fact_id]
    mem["suggested"] = [f for f in mem["suggested"] if f["id"] != fact_id]
    write_memory(username, mem)


def add_suggestions(username: str, texts: List[str]) -> List[Dict[str, Any]]:
    mem = read_memory(username)
    known = {f["text"].lower() for f in mem["facts"]}
    known |= {f["text"].lower() for f in mem["suggested"]}
    added = []
    for text in texts:
        if text.lower() in known:
            continue
        item = {"id": uuid.uuid4().hex[:8], "text": text, "source": "auto", "ts": time.time()}
        mem["suggested"].append(item)
        added.append(item)
        known.add(text.lower())
    mem["suggested"] = mem["suggested"][-20:]
    write_memory(username, mem)
    return added


def confirm_suggestion(username: str, fact_id: str) -> Optional[Dict[str, Any]]:
    mem = read_memory(username)
    for item in mem["suggested"]:
        if item["id"] == fact_id:
            mem["suggested"] = [s for s in mem["suggested"] if s["id"] != fact_id]
            item["source"] = "confirmed"
            mem["facts"].append(item)
            write_memory(username, mem)
            return item
    return None


# --- chat --------------------------------------------------------------------

def read_chat(username: str) -> List[Dict[str, Any]]:
    return _read_json(user_dir(username) / "chat.json", {"messages": []}).get("messages", [])


def append_chat(username: str, role: str, content: str) -> None:
    messages = read_chat(username)
    messages.append({"role": role, "content": content, "ts": time.time()})
    _write_json(user_dir(username) / "chat.json", {"messages": messages})


def clear_chat(username: str) -> None:
    _write_json(user_dir(username) / "chat.json", {"messages": []})
