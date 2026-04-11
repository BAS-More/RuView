//! REST API for WiFi-DensePose.
//!
//! Provides HTTP endpoints for health checks, pose queries, presence status,
//! and calibration management.

use axum::{
    extract::State,
    routing::get,
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use wifi_densepose_config::Config;

/// Shared application state.
///
/// Note: DataStore (rusqlite) is not Send+Sync, so DB access is handled
/// via blocking tasks or a separate connection per request.
pub struct AppState {
    pub config: Config,
    pub db_path: String,
    pub is_calibrated: bool,
}

/// Create the API router with all routes.
pub fn create_router(state: Arc<RwLock<AppState>>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/poses", get(get_poses))
        .route("/api/v1/presence", get(get_presence))
        .route("/api/v1/config", get(get_config))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "wifi-densepose-api",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn get_poses(
    State(state): State<Arc<RwLock<AppState>>>,
) -> Json<Value> {
    let s = state.read().await;
    let db_path = s.db_path.clone();
    drop(s);
    // Open a short-lived connection for the query
    match wifi_densepose_db::DataStore::open(&db_path) {
        Ok(db) => match db.recent_poses(10) {
            Ok(poses) => Json(json!({ "count": poses.len(), "poses": poses })),
            Err(e) => Json(json!({ "error": format!("{e}") })),
        },
        Err(e) => Json(json!({ "error": format!("{e}") })),
    }
}

async fn get_presence(
    State(state): State<Arc<RwLock<AppState>>>,
) -> Json<Value> {
    let s = state.read().await;
    let db_path = s.db_path.clone();
    drop(s);
    match wifi_densepose_db::DataStore::open(&db_path) {
        Ok(db) => match db.recent_presence(10) {
            Ok(events) => Json(json!({ "count": events.len(), "events": events })),
            Err(e) => Json(json!({ "error": format!("{e}") })),
        },
        Err(e) => Json(json!({ "error": format!("{e}") })),
    }
}

async fn get_config(
    State(state): State<Arc<RwLock<AppState>>>,
) -> Json<Value> {
    let s = state.read().await;
    Json(json!({
        "hardware": {
            "num_sensors": s.config.hardware.num_sensors,
            "num_subcarriers": s.config.hardware.num_subcarriers,
            "sampling_rate_hz": s.config.hardware.sampling_rate_hz,
        },
        "api": { "port": s.config.api.port },
        "is_calibrated": s.is_calibrated,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> Arc<RwLock<AppState>> {
        Arc::new(RwLock::new(AppState {
            config: Config::default(),
            db_path: ":memory:".to_string(),
            is_calibrated: false,
        }))
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::util::ServiceExt;

        let app = create_router(test_state());
        let response: axum::http::Response<_> = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_config_endpoint() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::util::ServiceExt;

        let app = create_router(test_state());
        let response: axum::http::Response<_> = app
            .oneshot(Request::builder().uri("/api/v1/config").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
