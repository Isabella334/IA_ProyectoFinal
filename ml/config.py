FEATURE_COLS = ["player_visible", "visibility_score", "player_distance", "noise_level", "paranoia_level", "has_investigation_target"]
TARGET_COL = "action"
TEST_SIZE  = 0.2
RANDOM_STATE = 42
RF_HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "min_samples_split": 20,
    "random_state": 42,
    "n_jobs": -1,
}
MODEL_OUTPUT_PATH = "models/npc_model.json"
