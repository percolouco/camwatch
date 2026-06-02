import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "/data/camwatch.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            camera TEXT,
            start_time INTEGER,
            snapshot_path TEXT,
            clip_path TEXT,
            plate TEXT,
            color_hex TEXT,
            color_name TEXT,
            processed_at INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS whitelist (
            plate TEXT PRIMARY KEY,
            label TEXT,
            added_at INTEGER
        )
    """)
    # Migrate existing DBs that lack clip_path
    try:
        conn.execute("ALTER TABLE events ADD COLUMN clip_path TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

def event_exists(event_id: str) -> bool:
    conn = get_db()
    row = conn.execute("SELECT id FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return row is not None

def insert_event(event_id, camera, start_time, snapshot_path, clip_path, plate, color_hex, color_name):
    import time
    conn = get_db()
    conn.execute("""
        INSERT OR IGNORE INTO events
        (id, camera, start_time, snapshot_path, clip_path, plate, color_hex, color_name, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_id, camera, start_time, snapshot_path, clip_path, plate, color_hex, color_name, int(time.time())))
    conn.commit()
    conn.close()

def get_events(limit=100, offset=0, plate_filter=None, camera_filter=None, date_filter=None):
    conn = get_db()
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if plate_filter:
        query += " AND plate LIKE ?"
        params.append(f"%{plate_filter}%")
    if camera_filter:
        query += " AND camera = ?"
        params.append(camera_filter)
    if date_filter:
        import time
        from datetime import datetime
        try:
            dt = datetime.strptime(date_filter, "%Y-%m-%d")
            ts_start = int(dt.timestamp())
            ts_end = ts_start + 86400
            query += " AND start_time >= ? AND start_time < ?"
            params.extend([ts_start, ts_end])
        except Exception:
            pass
    query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def count_events(plate_filter=None, camera_filter=None, date_filter=None):
    conn = get_db()
    query = "SELECT COUNT(*) FROM events WHERE 1=1"
    params = []
    if plate_filter:
        query += " AND plate LIKE ?"
        params.append(f"%{plate_filter}%")
    if camera_filter:
        query += " AND camera = ?"
        params.append(camera_filter)
    if date_filter:
        import time
        from datetime import datetime
        try:
            dt = datetime.strptime(date_filter, "%Y-%m-%d")
            ts_start = int(dt.timestamp())
            ts_end = ts_start + 86400
            query += " AND start_time >= ? AND start_time < ?"
            params.extend([ts_start, ts_end])
        except Exception:
            pass
    count = conn.execute(query, params).fetchone()[0]
    conn.close()
    return count

def get_event(event_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_event(event_id: str):
    conn = get_db()
    conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()


def update_plate(event_id: str, plate: str):
    conn = get_db()
    conn.execute("UPDATE events SET plate = ? WHERE id = ?", (plate.strip().upper() or None, event_id))
    conn.commit()
    conn.close()


def get_cameras():
    conn = get_db()
    rows = conn.execute("SELECT DISTINCT camera FROM events ORDER BY camera").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ── Whitelist ─────────────────────────────────────────────────────────────────

def get_whitelist() -> list[dict]:
    conn = get_db()
    rows = conn.execute("SELECT plate, label, added_at FROM whitelist ORDER BY added_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_whitelisted(plate: str) -> bool:
    if not plate:
        return False
    conn = get_db()
    row = conn.execute("SELECT plate FROM whitelist WHERE plate = ?", (plate.upper().strip(),)).fetchone()
    conn.close()
    return row is not None


def add_to_whitelist(plate: str, label: str = ""):
    import time
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO whitelist (plate, label, added_at) VALUES (?, ?, ?)",
        (plate.upper().strip(), label.strip(), int(time.time())),
    )
    conn.commit()
    conn.close()


def remove_from_whitelist(plate: str):
    conn = get_db()
    conn.execute("DELETE FROM whitelist WHERE plate = ?", (plate.upper().strip(),))
    conn.commit()
    conn.close()


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_plate_stats() -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT plate,
               COUNT(*)        AS count,
               MIN(start_time) AS first_seen,
               MAX(start_time) AS last_seen
        FROM events
        WHERE plate IS NOT NULL AND plate != ''
        GROUP BY plate
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
