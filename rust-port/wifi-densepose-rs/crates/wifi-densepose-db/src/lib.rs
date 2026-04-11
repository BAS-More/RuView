//! SQLite-based storage for WiFi-DensePose.
//!
//! Stores CSI frames, pose results, presence events, and calibration data.

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// A stored presence detection event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceEvent {
    pub id: Option<i64>,
    pub timestamp: f64,
    pub motion_level: String,
    pub confidence: f64,
    pub rssi_variance: f64,
}

/// A stored pose detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseRecord {
    pub id: Option<i64>,
    pub timestamp: f64,
    pub num_persons: usize,
    pub keypoints_json: String,
    pub confidence: f64,
}

/// SQLite-backed data store for RuView.
pub struct DataStore {
    conn: Connection,
}

impl DataStore {
    /// Open or create a database at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.init_tables()?;
        Ok(store)
    }

    /// Open an in-memory database (for testing).
    pub fn in_memory() -> Result<Self, DbError> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.init_tables()?;
        Ok(store)
    }

    fn init_tables(&self) -> Result<(), DbError> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS presence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                motion_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                rssi_variance REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pose_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                num_persons INTEGER NOT NULL,
                keypoints_json TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                noise_floor_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_presence_ts ON presence_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pose_ts ON pose_records(timestamp);"
        )?;
        Ok(())
    }

    /// Insert a presence detection event.
    pub fn insert_presence(&self, event: &PresenceEvent) -> Result<i64, DbError> {
        self.conn.execute(
            "INSERT INTO presence_events (timestamp, motion_level, confidence, rssi_variance)
             VALUES (?1, ?2, ?3, ?4)",
            params![event.timestamp, event.motion_level, event.confidence, event.rssi_variance],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Insert a pose detection record.
    pub fn insert_pose(&self, record: &PoseRecord) -> Result<i64, DbError> {
        self.conn.execute(
            "INSERT INTO pose_records (timestamp, num_persons, keypoints_json, confidence)
             VALUES (?1, ?2, ?3, ?4)",
            params![record.timestamp, record.num_persons as i64, record.keypoints_json, record.confidence],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Query recent presence events.
    pub fn recent_presence(&self, limit: usize) -> Result<Vec<PresenceEvent>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, motion_level, confidence, rssi_variance
             FROM presence_events ORDER BY timestamp DESC LIMIT ?1"
        )?;
        let events = stmt.query_map(params![limit as i64], |row| {
            Ok(PresenceEvent {
                id: Some(row.get(0)?),
                timestamp: row.get(1)?,
                motion_level: row.get(2)?,
                confidence: row.get(3)?,
                rssi_variance: row.get(4)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(events)
    }

    /// Query recent pose records.
    pub fn recent_poses(&self, limit: usize) -> Result<Vec<PoseRecord>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, num_persons, keypoints_json, confidence
             FROM pose_records ORDER BY timestamp DESC LIMIT ?1"
        )?;
        let records = stmt.query_map(params![limit as i64], |row| {
            Ok(PoseRecord {
                id: Some(row.get(0)?),
                timestamp: row.get(1)?,
                num_persons: row.get::<_, i64>(2)? as usize,
                keypoints_json: row.get(3)?,
                confidence: row.get(4)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(records)
    }

    /// Get count of presence events.
    pub fn presence_count(&self) -> Result<usize, DbError> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM presence_events", [], |row| row.get(0)
        )?;
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_query_presence() {
        let store = DataStore::in_memory().unwrap();
        let event = PresenceEvent {
            id: None,
            timestamp: 1700000000.0,
            motion_level: "active".to_string(),
            confidence: 0.95,
            rssi_variance: 2.5,
        };
        let id = store.insert_presence(&event).unwrap();
        assert!(id > 0);

        let events = store.recent_presence(10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].motion_level, "active");
    }

    #[test]
    fn test_create_and_query_pose() {
        let store = DataStore::in_memory().unwrap();
        let record = PoseRecord {
            id: None,
            timestamp: 1700000000.0,
            num_persons: 2,
            keypoints_json: "[]".to_string(),
            confidence: 0.8,
        };
        store.insert_pose(&record).unwrap();

        let poses = store.recent_poses(10).unwrap();
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].num_persons, 2);
    }
}
