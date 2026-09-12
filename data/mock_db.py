"""SQLite seed + access layer for customers/orders/tickets/appointments/escalations/refunds.

The store simulates orders placed through an Amazon-style storefront — order
IDs follow Amazon's public "NNN-NNNNNNN-NNNNNNN" shape and items are the kind
of products you'd actually find there. It's fictional data for a local demo;
nothing here talks to the real Amazon API or any of its services.

This module owns the schema, the connection, and the seed data only.
Domain-specific queries (e.g. "look up an order") belong in the tool that
needs them (agent/tools/orders.py, Phase 1) — keeping this file a dumb data
layer is what lets those tools stay simple, deterministic functions.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "mock_data.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    phone       TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id            TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL,
    item                TEXT NOT NULL,
    quantity            INTEGER NOT NULL DEFAULT 1,
    price               REAL NOT NULL,
    status              TEXT NOT NULL,
    order_date          TEXT NOT NULL,
    estimated_delivery  TEXT,
    tracking_number     TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id       TEXT NOT NULL,
    issue             TEXT NOT NULL,
    resolution        TEXT,
    sentiment         TEXT,
    follow_up_needed  INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE IF NOT EXISTS appointments (
    appointment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     TEXT NOT NULL,
    scheduled_time  TEXT NOT NULL,
    reason          TEXT,
    status          TEXT NOT NULL DEFAULT 'scheduled',
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- Phase 4: a handoff packet, created whenever the escalation tracker in
-- agent/tools/escalation.py decides a human needs to take over. "For now,
-- transfer to human just logs the packet" (PROJECT_PLAN.md) — this table is
-- that log; a real warm transfer arrives in Phase 10.
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id            TEXT NOT NULL,
    reason                 TEXT NOT NULL,
    customer_intent        TEXT NOT NULL,
    conversation_summary   TEXT NOT NULL,
    verified_account_info  TEXT,
    actions_taken          TEXT,
    sentiment              TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    -- Phase 11: whether create_handoff_packet's call to notify_escalation()
    -- (agent/tools/notifications.py) actually delivered this packet to the
    -- configured automation platform. Defaults to unnotified; a session
    -- with no ESCALATION_WEBHOOK_URL configured leaves every row this way.
    notified                INTEGER NOT NULL DEFAULT 0,
    notified_at             TEXT,
    -- Phase 12: a handover is a process, not an event. `items` is a
    -- newline-separated list of every reason on this ONE handover — one
    -- customer gets one callback, so a second escalation-worthy issue appends
    -- here rather than opening a second row. Reasons are written by this
    -- codebase, never by a customer, so none can contain a newline.
    -- `resolution` is null while the handover is open, which is also how an
    -- abandoned call is recognised at close_session.
    items                   TEXT,
    resolution              TEXT,
    callback_time           TEXT,
    resolved_at             TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- Phase 6: one row per issued refund. No seed data on purpose — adding a
-- sample row would mean retroactively marking one of the ORDERS rows above
-- "Refunded", which risks breaking other phases' tests that reference
-- specific seeded orders by status. Every refunds test builds its own
-- isolated DB anyway, matching this project's existing convention.
CREATE TABLE IF NOT EXISTS refunds (
    refund_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT NOT NULL,
    customer_id     TEXT NOT NULL,
    amount          REAL NOT NULL,
    item_condition  TEXT NOT NULL,
    reason          TEXT,
    issued_at       TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);
"""

# customer_id, name, email, phone
CUSTOMERS = [
    ("CUST-1001", "Maria Gonzalez", "maria.gonzalez@example.com", "+1-555-0101"),
    ("CUST-1002", "James Whitfield", "james.whitfield@example.com", "+1-555-0102"),
    ("CUST-1003", "Priya Natarajan", "priya.natarajan@example.com", "+1-555-0103"),
    ("CUST-1004", "Tom O'Brien", "tom.obrien@example.com", "+1-555-0104"),
    ("CUST-1005", "Aiko Tanaka", "aiko.tanaka@example.com", "+1-555-0105"),
]

# order_id, customer_id, item, quantity, price, status, order_date, estimated_delivery, tracking_number
# order_id format mirrors Amazon's real "NNN-NNNNNNN-NNNNNNN" order numbers (fictional values).
#
# THESE DATES NEED PERIODIC REFRESHING, and here is why. issue_refund checks
# estimated_delivery against a 30-day return window using the real clock
# (agent/tools/refunds.py), so a Delivered order silently ages out of
# eligibility as wall-clock time passes. When that happens the refund flow
# stops being demonstrable and the live refund tests fail for a *calendar*
# reason that reads exactly like a logic regression — which is precisely what
# happened on 2026-09-01, undetected for a week because an invalid API key was
# making those tests fail for a different reason at the same time.
#
# In production this would never arise: real orders keep arriving, so there is
# always something inside the window. A fixed seed has no such supply, so the
# convention is to shift these dates forward whenever they get stale. Last
# refreshed 2026-09-08 (shifted +18 days). The rule: every "Delivered" order's
# estimated_delivery stays within 30 days of today, in-transit orders keep
# estimated_delivery in the near future, and no order_date is later than today.
#
# Phase 10c removes the need to keep doing this for the *tests* — its eval
# harness freezes the clock to each recording's timestamp, so a recorded
# scenario evaluates against the dates that were true when it was recorded.
# Refreshing will still matter for live demos.
ORDERS = [
    ("112-3487561-2938471", "CUST-1001", "Echo Dot (5th Gen, Charcoal)", 1, 34.99,
     "Delivered", "2026-08-28", "2026-08-31", "TBA123456789US"),
    ("113-9284756-1029384", "CUST-1001", "Kindle Paperwhite (16 GB)", 1, 139.99,
     "Out for delivery", "2026-09-06", "2026-09-09", "TBA987654321US"),
    ("114-2938475-6193847", "CUST-1002", "Anker 6-in-1 USB-C Hub", 2, 25.99,
     "Shipped", "2026-09-05", "2026-09-11", "TBA564738291US"),
    ("115-4857392-8374651", "CUST-1002", "Instant Pot Duo 7-in-1 (6 Qt)", 1, 89.00,
     "Processing", "2026-09-08", "2026-09-14", None),
    ("116-1029384-7563829", "CUST-1003", "Nike Air Zoom Pegasus 40, Size 9", 1, 129.95,
     "Delivered", "2026-08-23", "2026-08-26", "TBA192837465US"),
    ("117-6748291-3049582", "CUST-1003", "Logitech MX Master 3S Mouse", 1, 99.99,
     "Cancelled", "2026-09-02", None, None),
    ("118-8374659-2019384", "CUST-1004", "Stanley Quencher 40oz Tumbler", 1, 45.00,
     "Delayed", "2026-08-30", "2026-09-12", "TBA827364519US"),
    ("119-5647382-9182736", "CUST-1005", "Sony WH-1000XM5 Headphones", 1, 349.99,
     "Delivered", "2026-08-17", "2026-08-20", "TBA736451928US"),
]

# customer_id, issue, resolution, sentiment, follow_up_needed, created_at
TICKETS = [
    ("CUST-1003", "Received wrong color for order 116-1029384-7563829 sneakers",
     "Replacement shipped, no charge", "neutral", 0, "2026-08-27"),
]

# customer_id, scheduled_time, reason, status
APPOINTMENTS = [
    ("CUST-1004", "2026-09-12T15:00:00", "Callback re: delayed order 118-8374659-2019384", "scheduled"),
]

# customer_id, reason, customer_intent, conversation_summary, verified_account_info,
# actions_taken, sentiment, created_at
ESCALATIONS = [
    ("CUST-1002", "explicit request for a human",
     "Wanted to speak with a person about a recurring billing problem",
     "Customer asked for a human agent after describing a billing issue that had come up twice before.",
     "Customer ID CUST-1002", "None — handed off before any tool was used.",
     "negative", "2026-08-20T10:15:00"),
]


@contextmanager
def get_connection():
    """Yield a sqlite3 connection with foreign keys enforced and dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(reset: bool = False) -> None:
    """Create the schema. If reset, drop all tables first (child tables before parents)."""
    with get_connection() as conn:
        if reset:
            conn.executescript(
                "DROP TABLE IF EXISTS refunds;"
                "DROP TABLE IF EXISTS escalations;"
                "DROP TABLE IF EXISTS tickets;"
                "DROP TABLE IF EXISTS appointments;"
                "DROP TABLE IF EXISTS orders;"
                "DROP TABLE IF EXISTS customers;"
            )
        conn.executescript(SCHEMA)


def seed_db() -> None:
    """Populate the tables with fake Amazon-style data. Idempotent: no-ops if already seeded."""
    with get_connection() as conn:
        already_seeded = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if already_seeded:
            return
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", ORDERS)
        conn.executemany(
            "INSERT INTO tickets "
            "(customer_id, issue, resolution, sentiment, follow_up_needed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            TICKETS,
        )
        conn.executemany(
            "INSERT INTO appointments (customer_id, scheduled_time, reason, status) "
            "VALUES (?, ?, ?, ?)",
            APPOINTMENTS,
        )
        conn.executemany(
            "INSERT INTO escalations "
            "(customer_id, reason, customer_intent, conversation_summary, "
            "verified_account_info, actions_taken, sentiment, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ESCALATIONS,
        )


def reset_and_seed() -> None:
    """Wipe and reseed from scratch — used by tests and by running this file directly."""
    init_db(reset=True)
    seed_db()


if __name__ == "__main__":
    reset_and_seed()
    print(f"Seeded mock DB at {DB_PATH}")
