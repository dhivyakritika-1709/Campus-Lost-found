"""
database.py
SQLite database management for Campus Lost & Found System.
Handles persistent storage of reports and secure contact requests.
Strictly protects private contact information using parameterized queries.
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "lost_found.db")


def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes tables and seeds initial data if database is empty."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL CHECK(report_type IN ('LOST', 'FOUND')),
            item_name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            colour TEXT,
            location TEXT,
            date TEXT,
            identifying_details TEXT,
            reporter_name TEXT NOT NULL,
            contact_number TEXT NOT NULL,
            image_path TEXT,
            status TEXT DEFAULT 'ACTIVE',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Contact Requests table for privacy-preserving owner/finder communication
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contact_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requester_report_id INTEGER NOT NULL,
            target_report_id INTEGER NOT NULL,
            requester_message TEXT,
            status TEXT DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'ACCEPTED', 'DECLINED')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (requester_report_id) REFERENCES reports (id),
            FOREIGN KEY (target_report_id) REFERENCES reports (id)
        )
    """)

    # 3. Persistent successful connections counter.
    # This counter is intentionally independent of reports/contact request rows,
    # so deleting a report does not reduce the displayed total.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app_stats (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        INSERT OR IGNORE INTO app_stats (key, value)
        SELECT 'successful_connections', COUNT(*)
        FROM contact_requests
        WHERE status = 'ACCEPTED'
    """)

    conn.commit()

    # Check if empty, then seed sample records for testing
    cursor.execute("SELECT COUNT(*) as cnt FROM reports")
    count = cursor.fetchone()["cnt"]
    if count == 0:
        _seed_sample_records(conn)

    conn.close()


def _seed_sample_records(conn):
    """Seeds realistic sample campus Lost and Found reports."""
    cursor = conn.cursor()
    sample_reports = [
        # Pair 1: Water bottle (High match)
        (
            "LOST",
            "Black Milton Water Bottle",
            "Daily Essentials",
            "Stainless steel 1-litre insulated flask with a minor scratch on the cap",
            "Black",
            "Central Library 2nd Floor",
            "2026-09-18",
            "Dent on bottom rim, white sticker on bottom",
            "Aarav Sharma",
            "+91 98765 43210",
            None,
            "ACTIVE"
        ),
        (
            "FOUND",
            "Black Stainless Steel Water Flask",
            "Daily Essentials",
            "Milton insulated water bottle found near study carrels",
            "Black",
            "Central Library Reading Room",
            "2026-09-18",
            "Found near desk #42 with scratched cap",
            "Campus Security - Officer Verma",
            "+91 91234 56789",
            None,
            "ACTIVE"
        ),
        # Pair 2: College ID Card (High match)
        (
            "LOST",
            "Student ID Card - CSE Dept",
            "Campus Specific",
            "Blue lanyard with University RFID Smart Card. Roll number ending in 042",
            "Blue",
            "Cafeteria / Food Court",
            "2026-09-19",
            "B.Tech CSE Year 2 ID card, key attached to ring",
            "Dharshini K",
            "+91 98111 22334",
            None,
            "ACTIVE"
        ),
        (
            "FOUND",
            "College Student ID with Lanyard",
            "Campus Specific",
            "Blue tag university identity card for 2nd year Computer Science student",
            "Blue",
            "Main Canteen Counter",
            "2026-09-20",
            "Small silver locker key attached to lanyard ring",
            "Rohan Gupta",
            "+91 97222 33445",
            None,
            "ACTIVE"
        ),
        # Pair 3: Laptop Charger
        (
            "LOST",
            "HP 65W USB-C Laptop Charger",
            "Electronics",
            "Black HP power adapter with braided cable and Velcro tie",
            "Black",
            "Computer Lab 4 (Turing Hall)",
            "2026-09-21",
            "Small piece of red electrical tape near connector",
            "Kavya Nair",
            "+91 99445 56677",
            None,
            "ACTIVE"
        ),
        # Pair 4: Umbrella (Non-match to charger, match to something else)
        (
            "FOUND",
            "Foldable Navy Blue Umbrella",
            "Daily Essentials",
            "Three-fold compact umbrella left on the umbrella rack",
            "Blue",
            "Academic Block B Entrance",
            "2026-09-21",
            "Wooden curved handle",
            "Aditya Mehta",
            "+91 93333 44455",
            None,
            "ACTIVE"
        )
    ]

    cursor.executemany("""
        INSERT INTO reports (
            report_type, item_name, category, description, colour,
            location, date, identifying_details, reporter_name,
            contact_number, image_path, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_reports)

    # Seed one initial contact request from Lost report #1 to Found report #2
    cursor.execute("""
        INSERT INTO contact_requests (
            requester_report_id, target_report_id, requester_message, status
        ) VALUES (1, 2, 'Hi, I believe this is my Milton water bottle lost on 18th Sept. Could we coordinate?', 'PENDING')
    """)

    conn.commit()


def create_report(report_type, item_name, category, description, colour,
                  location, date, identifying_details, reporter_name,
                  contact_number, image_path=None):
    """
    Creates a new report using parameterized queries.
    Returns the newly created report ID.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (
            report_type, item_name, category, description, colour,
            location, date, identifying_details, reporter_name,
            contact_number, image_path, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
    """, (
        report_type.strip().upper(),
        item_name.strip(),
        category.strip(),
        description.strip() if description else "",
        colour.strip() if colour else "",
        location.strip() if location else "",
        date.strip() if date else datetime.now().strftime("%Y-%m-%d"),
        identifying_details.strip() if identifying_details else "",
        reporter_name.strip(),
        contact_number.strip(),
        image_path
    ))
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id


def get_report_by_id(report_id, include_private=False):
    """
    Fetches a single report by ID.
    If include_private is False, contact_number and identifying_details are sanitized.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    data = dict(row)
    if not include_private:
        data["contact_number"] = "[Private - Request Contact to view]"
        data["identifying_details"] = "[Confidential verification details]"
    return data


def get_reports_for_matching(opposite_type):
    """
    Returns active reports of the opposite type for ML matching.
    Includes feature fields, while keeping contact numbers hidden from ML inference.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, report_type, item_name, category, description, colour,
               location, date, reporter_name, image_path, created_at
        FROM reports
        WHERE report_type = ? AND status = 'ACTIVE'
        ORDER BY id DESC
    """, (opposite_type.upper(),))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_reports(report_type=None, status="ACTIVE", search_query=None, category=None):
    """
    Returns public-safe reports.
    CONTACT NUMBER IS STRICTLY EXCLUDED to guarantee student privacy.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT id, report_type, item_name, category, description, colour,
               location, date, reporter_name, image_path, status, created_at
        FROM reports
        WHERE 1=1
    """
    params = []

    if report_type and report_type != "ALL":
        query += " AND report_type = ?"
        params.append(report_type.upper())

    if status and status != "ALL":
        query += " AND status = ?"
        params.append(status)

    if category and category != "All Categories":
        query += " AND category = ?"
        params.append(category)

    if search_query:
        query += " AND (item_name LIKE ? OR description LIKE ? OR location LIKE ? OR colour LIKE ?)"
        like_term = f"%{search_query.strip()}%"
        params.extend([like_term, like_term, like_term, like_term])

    query += " ORDER BY id DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_reports_by_user_query(query_str):
    """
    Finds reports matching a user query (Report ID, Reporter Name, or Phone Number).
    Allows students to locate and manage their own reports in the 'My Reports' view.
    """
    if not query_str:
        return []

    conn = get_connection()
    cursor = conn.cursor()
    query_clean = str(query_str).strip()

    # Try numeric ID first
    if query_clean.isdigit():
        cursor.execute("SELECT * FROM reports WHERE id = ?", (int(query_clean),))
        rows = cursor.fetchall()
        if rows:
            conn.close()
            return [dict(r) for r in rows]

    # Otherwise search by name or contact number
    cursor.execute("""
        SELECT * FROM reports
        WHERE reporter_name LIKE ? OR contact_number LIKE ?
        ORDER BY id DESC
    """, (f"%{query_clean}%", f"%{query_clean}%"))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_contact_request(requester_report_id, target_report_id, requester_message=""):
    """
    Creates a new contact request between two reports.
    Validates that:
    1. The reports exist and are opposite types.
    2. A pending/accepted request doesn't already exist.
    """
    if requester_report_id == target_report_id:
        return False, "You cannot request contact with your own report."

    conn = get_connection()
    cursor = conn.cursor()

    # Check existence & types
    cursor.execute("SELECT id, report_type FROM reports WHERE id = ?", (requester_report_id,))
    req_row = cursor.fetchone()
    cursor.execute("SELECT id, report_type FROM reports WHERE id = ?", (target_report_id,))
    target_row = cursor.fetchone()

    if not req_row or not target_row:
        conn.close()
        return False, "One or both reports do not exist."

    if req_row["report_type"] == target_row["report_type"]:
        conn.close()
        return False, "Cannot send a contact request between two reports of the same type."

    # Check duplicate
    cursor.execute("""
        SELECT id, status FROM contact_requests
        WHERE requester_report_id = ? AND target_report_id = ?
    """, (requester_report_id, target_report_id))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return False, f"A contact request already exists with status: {existing['status']}."

    cursor.execute("""
        INSERT INTO contact_requests (requester_report_id, target_report_id, requester_message, status)
        VALUES (?, ?, ?, 'PENDING')
    """, (requester_report_id, target_report_id, requester_message.strip()))
    conn.commit()
    conn.close()
    return True, "Contact request sent successfully! You will be notified once reviewed."


def get_contact_requests_for_user(report_ids):
    """
    Returns all incoming and outgoing contact requests for a list of report IDs.
    Applies privacy rules:
    - If status == 'ACCEPTED', both parties' contact numbers are revealed!
    - If status != 'ACCEPTED', contact numbers remain private.
    """
    if not report_ids:
        return {"incoming": [], "outgoing": []}

    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ",".join("?" for _ in report_ids)

    # Incoming requests: someone requested contact with one of my reports (target_report_id IN report_ids)
    cursor.execute(f"""
        SELECT cr.id as request_id, cr.status, cr.requester_message, cr.created_at, cr.updated_at,
               cr.requester_report_id, cr.target_report_id,
               r_req.item_name as req_item_name, r_req.report_type as req_type,
               r_req.reporter_name as req_reporter_name, r_req.contact_number as req_contact_number,
               r_tgt.item_name as tgt_item_name, r_tgt.report_type as tgt_type,
               r_tgt.reporter_name as tgt_reporter_name, r_tgt.contact_number as tgt_contact_number
        FROM contact_requests cr
        JOIN reports r_req ON cr.requester_report_id = r_req.id
        JOIN reports r_tgt ON cr.target_report_id = r_tgt.id
        WHERE cr.target_report_id IN ({placeholders})
        ORDER BY cr.id DESC
    """, report_ids)
    incoming_rows = [dict(r) for r in cursor.fetchall()]

    # Outgoing requests: I requested contact with someone else's report (requester_report_id IN report_ids)
    cursor.execute(f"""
        SELECT cr.id as request_id, cr.status, cr.requester_message, cr.created_at, cr.updated_at,
               cr.requester_report_id, cr.target_report_id,
               r_req.item_name as req_item_name, r_req.report_type as req_type,
               r_req.reporter_name as req_reporter_name, r_req.contact_number as req_contact_number,
               r_tgt.item_name as tgt_item_name, r_tgt.report_type as tgt_type,
               r_tgt.reporter_name as tgt_reporter_name, r_tgt.contact_number as tgt_contact_number
        FROM contact_requests cr
        JOIN reports r_req ON cr.requester_report_id = r_req.id
        JOIN reports r_tgt ON cr.target_report_id = r_tgt.id
        WHERE cr.requester_report_id IN ({placeholders})
        ORDER BY cr.id DESC
    """, report_ids)
    outgoing_rows = [dict(r) for r in cursor.fetchall()]

    conn.close()

    # Enforce privacy masking based on status
    for req in incoming_rows:
        if req["status"] != "ACCEPTED":
            req["req_contact_number"] = "[Revealed upon acceptance]"

    for req in outgoing_rows:
        if req["status"] != "ACCEPTED":
            req["tgt_contact_number"] = "[Revealed when owner/finder accepts]"

    return {"incoming": incoming_rows, "outgoing": outgoing_rows}


def accept_contact_request(request_id):
    """Accepts a contact request, allowing both parties to view contact numbers."""
    conn = get_connection()
    cursor = conn.cursor()

    # Only a newly accepted request creates a new successful connection.
    cursor.execute("""
        SELECT status FROM contact_requests
        WHERE id = ?
    """, (request_id,))
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return False

    if row["status"] != "ACCEPTED":
        cursor.execute("""
            UPDATE contact_requests
            SET status = 'ACCEPTED', updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (request_id,))

        cursor.execute("""
            UPDATE app_stats
            SET value = value + 1
            WHERE key = 'successful_connections'
        """)

    conn.commit()
    conn.close()
    return True


def decline_contact_request(request_id):
    """Declines a contact request, permanently protecting phone numbers."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE contact_requests
        SET status = 'DECLINED', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (request_id,))
    conn.commit()
    conn.close()
    return True


def delete_report(report_id):
    """
    Permanently deletes a report and all contact requests linked to it.

    This removes the report from the report registry, matching pool,
    and public reports. Returns (success, message).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Confirm the report exists and capture its image path before deletion.
    cursor.execute("SELECT image_path FROM reports WHERE id = ?", (report_id,))
    report = cursor.fetchone()

    if not report:
        conn.close()
        return False, "Report not found."

    image_path = report["image_path"]

    # Remove all contact requests involving this report first.
    cursor.execute("""
        DELETE FROM contact_requests
        WHERE requester_report_id = ? OR target_report_id = ?
    """, (report_id, report_id))

    # Permanently remove the report itself.
    cursor.execute("DELETE FROM reports WHERE id = ?", (report_id,))

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    # Remove the uploaded image belonging to this report, if any.
    if deleted and image_path:
        try:
            if os.path.isfile(image_path):
                os.remove(image_path)
        except OSError:
            # Database deletion is still successful even if an old image
            # file cannot be removed from disk.
            pass

    if deleted:
        return True, "Report deleted successfully."

    return False, "Report could not be deleted."


def update_report_status(report_id, new_status):
    """Updates report status (e.g. 'RESOLVED' once handed over)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE reports SET status = ? WHERE id = ?", (new_status, report_id))
    conn.commit()
    conn.close()
    return True


def get_stats():
    """Returns real-time aggregate statistics for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total_lost FROM reports WHERE report_type = 'LOST'")
    total_lost = cursor.fetchone()["total_lost"]

    cursor.execute("SELECT COUNT(*) as total_found FROM reports WHERE report_type = 'FOUND'")
    total_found = cursor.fetchone()["total_found"]

    cursor.execute("SELECT COUNT(*) as resolved FROM reports WHERE status = 'RESOLVED'")
    resolved = cursor.fetchone()["resolved"]

    cursor.execute("SELECT COUNT(*) as pending_requests FROM contact_requests WHERE status = 'PENDING'")
    pending_requests = cursor.fetchone()["pending_requests"]

    cursor.execute("""
        SELECT value AS successful_connections
        FROM app_stats
        WHERE key = 'successful_connections'
    """)
    row = cursor.fetchone()
    accepted_requests = row["successful_connections"] if row else 0

    conn.close()

    return {
        "total_lost": total_lost,
        "total_found": total_found,
        "total_reports": total_lost + total_found,
        "resolved": resolved,
        "pending_requests": pending_requests,
        "accepted_requests": accepted_requests
    }


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
    stats = get_stats()
    print("Current stats:", stats)
