import json
import os
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path(__file__).parent.parent / "data" / "history"

def _ensure_history_dir():
    if not HISTORY_DIR.exists():
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def save_report(topic: str, final_state: dict, logs: list):
    """Save a generated report state and logs to a JSON file."""
    _ensure_history_dir()
    
    # Generate a safe filename
    safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:50].strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{safe_topic}_{file_id}.json"
    
    file_path = HISTORY_DIR / filename
    
    data = {
        "id": file_id,
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "final_state": final_state,
        "logs": logs
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return str(file_path)

def list_past_reports():
    """Return a list of metadata for all saved reports, sorted by newest first."""
    _ensure_history_dir()
    reports = []
    
    for file_path in HISTORY_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                reports.append({
                    "id": data.get("id"),
                    "filename": file_path.name,
                    "timestamp": data.get("timestamp"),
                    "topic": data.get("topic", "Unknown Topic")
                })
        except Exception:
            pass # Skip corrupted files
            
    # Sort by timestamp descending
    return sorted(reports, key=lambda x: x["timestamp"], reverse=True)

def load_report(filename: str):
    """Load a specific report by filename."""
    file_path = HISTORY_DIR / filename
    if not file_path.exists():
        return None
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
