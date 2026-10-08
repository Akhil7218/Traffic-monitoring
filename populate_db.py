"""
TrafficSentinel AI — Database Seeding & Mock Data Generator
Populates violations.db with realistic sample traffic infraction logs for demonstration & dashboard analytics testing.
"""

import sqlite3
import os
import random
from datetime import datetime, timedelta
import config

DB_PATH = config.DB_PATH

SAMPLE_PLATES = [
    "KA03MX4521", "MH12AB3456", "DL09WR6392", "TN05AT7024",
    "KL07CD5678", "UP32GH8901", "RJ14XY2345", "GJ01BC7890",
    "TS09QR1234", "KA01HJ9876", "MH04CD1234", "DL08PQ5678",
    "KL09CA1671", "TN22EF3456", "KA05MN7654", "MH20ST9012",
]

SAMPLE_OWNERS = [
    "Rajesh Kumar",    "Priya Sharma",   "Mohammed Irfan",
    "Sunita Patel",    "Amit Verma",     "Deepa Nair",
    "Suresh Reddy",    "Anita Joshi",    "Vikram Singh",
    "Kavitha Menon",   "Ravi Teja",      "Pooja Gupta",
    "UNKNOWN",         "UNKNOWN",        "Arjun Das",
    "Meena Krishnan",
]

SAMPLE_VIDEOS = [
    "camera_traffic_01.mp4", "camera_traffic_02.mp4", "camera_main_road.mp4"
]

INFRACTION_TYPES = [
    ("NO HELMET",              1000),
    ("TRIPLE RIDING",          1000),
    ("WRONG WAY",              5000),
    ("NO HELMET + TRIPLE RIDING", 2000),
]


def generate_mock_records(count=25):
    records = []
    current_time = datetime.now()
    for _ in range(count):
        days_offset = random.choices([0, 1, 2, 3, 4, 5, 6], weights=[8, 6, 5, 4, 3, 2, 1])[0]
        hour = random.randint(7, 22)
        minute = random.randint(0, 59)
        timestamp = (current_time - timedelta(days=days_offset)).replace(
            hour=hour, minute=minute, second=random.randint(0, 59)
        )

        idx = random.randint(0, len(SAMPLE_PLATES) - 1)
        plate = SAMPLE_PLATES[idx]
        owner = SAMPLE_OWNERS[idx]
        infraction, base_fee = random.choice(INFRACTION_TYPES)

        prior_count = sum(1 for r in records if r["plate"] == plate)
        multiplier = min(prior_count + 1, 3)
        total_fine = base_fee * multiplier

        is_paid = 1 if (days_offset >= 3 and random.random() < 0.45) else 0
        video_src = random.choice(SAMPLE_VIDEOS)

        records.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "video": video_src,
            "violation": infraction,
            "plate": plate,
            "owner_name": owner,
            "fine": total_fine,
            "paid": is_paid,
        })

    records.sort(key=lambda r: r["timestamp"])
    return records


def seed_database():
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA journal_mode=WAL")
    cursor = connection.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS violations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT,
        video       TEXT,
        violation   TEXT,
        plate       TEXT,
        owner_name  TEXT,
        fine        INTEGER,
        screenshot  TEXT,
        challan     TEXT,
        paid        INTEGER DEFAULT 0
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS visitors (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ip        TEXT,
        page      TEXT,
        referrer  TEXT,
        ua        TEXT
    )''')

    cursor.execute("DELETE FROM violations")
    connection.commit()

    records = generate_mock_records(25)
    for r in records:
        cursor.execute(
            "INSERT INTO violations "
            "(timestamp,video,violation,plate,owner_name,fine,screenshot,challan,paid) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (r["timestamp"], r["video"], r["violation"],
             r["plate"], r["owner_name"], r["fine"],
             None, None, r["paid"])
        )

    connection.commit()
    connection.close()

    print(f"✅ TrafficSentinel AI — Database seeded with {len(records)} records at {DB_PATH}")


if __name__ == "__main__":
    seed_database()
