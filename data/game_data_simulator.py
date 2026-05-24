"""
data/game_data_simulator.py

Simula cómo se verían los datos del NPC en condiciones reales de juego,
introduciendo imperfecciones de sensores alrededor de los umbrales de decisión.

Por qué existe este archivo
────────────────────────────
Los datos originales (training_data.csv) vienen de Godot con valores perfectos:
cada feature está limpiamente separada entre estados. Eso produce accuracy=1.0,
lo cual no refleja lo que pasaría con sensores ruidosos o situaciones ambiguas.

Este simulador replica la física del juego (los mismos rangos y umbrales reales)
pero agrega incertidumbre donde un NPC real la tendría:
  - ¿Escuché algo o fue el viento?         → noise cerca de 0.0025
  - ¿Lo vi o fue una sombra?               → visibility cerca de 0.4998
  - ¿Estoy seguro de perseguirlo aún?      → paranoia cerca de 0.2996

Cómo portarlo a Godot
──────────────────────
Cada función gen_* representa un estado del NPC. La lógica es:
  1. Tomar los valores base (lo que Godot mediría perfectamente)
  2. Agregar sensor_noise() — fluctuación gaussiana del sensor
  3. Agregar transition_blur() — incertidumbre cerca de umbrales de decisión

En GDScript equivaldría a:
  var noise_reading = raw_noise + randfn(0, SENSOR_SIGMA)
  var paranoia_reading = raw_paranoia + randfn(0, PARANOIA_SIGMA)
  if abs(paranoia_reading - PARANOIA_THRESHOLD) < BLUR_ZONE:
      paranoia_reading += randfn(0, BLUR_SIGMA)

Umbrales reales extraídos del npc_model.json
─────────────────────────────────────────────
  NOISE_THRESHOLD      = 0.0025   (patrol/alert vs investigate/chase)
  VISIBILITY_THRESHOLD = 0.4998   (investigate vs chase)
  PARANOIA_THRESHOLD   = 0.2996   (patrol vs alert)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Semilla y salida ──────────────────────────────────────────────────────────
RNG  = np.random.default_rng(42)
OUT  = Path("data/training_data_sim.csv")
N_PER_STATE = 800   # filas por estado (balanceado)

# ── Umbrales reales del modelo (extraídos de npc_model.json) ─────────────────
NOISE_THRESHOLD      = 0.0025
VISIBILITY_THRESHOLD = 0.4998
PARANOIA_THRESHOLD   = 0.2996

# ── Parámetros de ruido de sensores ──────────────────────────────────────────
# Estos son los valores que habría que calibrar en Godot
SENSOR_SIGMA    = 0.015   # ruido base de cualquier sensor continuo
BLUR_ZONE       = 0.08    # distancia al umbral donde se agrega incertidumbre extra
BLUR_SIGMA      = 0.04    # magnitud de esa incertidumbre extra
FLIP_PROB       = 0.05    # probabilidad de que un flag booleano sea incorrecto


# ── Funciones de ruido reutilizables ─────────────────────────────────────────
def sensor_noise(values: np.ndarray, sigma: float = SENSOR_SIGMA) -> np.ndarray:
    """
    Fluctuación gaussiana del sensor.
    Equivale a: value + randfn(0, sigma) en GDScript.
    """
    return values + RNG.normal(0, sigma, len(values))


def transition_blur(values: np.ndarray, threshold: float,
                    blur_zone: float = BLUR_ZONE,
                    blur_sigma: float = BLUR_SIGMA) -> np.ndarray:
    """
    Incertidumbre extra cuando el valor está cerca de un umbral de decisión.
    El NPC duda más cuando está justo en el límite entre dos estados.
    Equivale a en GDScript:
        if abs(value - threshold) < blur_zone:
            value += randfn(0, blur_sigma)
    """
    near = np.abs(values - threshold) < blur_zone
    blur = RNG.normal(0, blur_sigma, len(values))
    return np.where(near, values + blur, values)


def boolean_flip(values: np.ndarray, prob: float = FLIP_PROB) -> np.ndarray:
    """
    Probabilidad de que un flag booleano sea leído incorrectamente.
    Simula: sensor de visión que falla, oclusión momentánea, etc.
    Equivale a en GDScript:
        if randf() < flip_prob: value = 1.0 - value
    """
    flips = RNG.random(len(values)) < prob
    return np.where(flips, 1.0 - values, values)


def clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


# ── Generadores por estado ────────────────────────────────────────────────────
def gen_patrol(n: int) -> pd.DataFrame:
    """
    NPC patrullando. Sin estímulos activos.
    Rangos base del dataset real: paranoia [0, 0.30), noise=0, visible=0.
    La zona de transición patrol/alert vive en paranoia ~ 0.2996.
    """
    # Paranoia: mayormente baja, con cola hacia el umbral
    paranoia = RNG.uniform(0.0, PARANOIA_THRESHOLD, n)
    paranoia = sensor_noise(paranoia, sigma=0.01)
    paranoia = transition_blur(paranoia, threshold=PARANOIA_THRESHOLD)
    paranoia = clip01(paranoia)

    # Noise: perfectamente 0 en Godot, pero el micrófono tiene hiss
    noise = np.abs(RNG.normal(0, 0.003, n))          # ruido de fondo muy bajo
    noise = transition_blur(noise, threshold=NOISE_THRESHOLD)
    noise = clip01(noise)

    # Distancia: rango real patrol [31.3, 74.7]
    distance = RNG.uniform(31.3, 74.7, n)
    distance = sensor_noise(distance, sigma=2.0)
    distance = np.clip(distance, 10.0, 300.0)

    # Visibility: baja, proporcional a distancia
    visibility = clip01(1.0 - distance / 200.0 + RNG.normal(0, 0.02, n))
    visibility = transition_blur(visibility, threshold=VISIBILITY_THRESHOLD)

    player_visible = np.zeros(n)
    has_target     = np.zeros(n)

    return pd.DataFrame({
        "player_visible":           boolean_flip(player_visible),
        "visibility_score":         visibility.round(4),
        "player_distance":          distance.round(4),
        "noise_level":              noise.round(4),
        "paranoia_level":           paranoia.round(4),
        "has_investigation_target": boolean_flip(has_target),
        "action": "patrol",
    })


def gen_alert(n: int) -> pd.DataFrame:
    """
    NPC alerta. Paranoia acumulada pero sin estímulo activo ahora.
    Rangos base: paranoia [0.30, 1.0), noise=0, visible=0.
    La zona gris con patrol vive en paranoia ~ 0.2996.
    """
    # Paranoia: arriba del umbral, con solapamiento hacia patrol
    paranoia = RNG.uniform(PARANOIA_THRESHOLD, 1.0, n)
    paranoia = sensor_noise(paranoia, sigma=0.015)
    paranoia = transition_blur(paranoia, threshold=PARANOIA_THRESHOLD)
    paranoia = clip01(paranoia)

    # Noise: cero en Godot pero con hiss del sensor
    noise = np.abs(RNG.normal(0, 0.004, n))
    noise = transition_blur(noise, threshold=NOISE_THRESHOLD)
    noise = clip01(noise)

    # Distancia: rango real alert [31.3, 74.7]
    distance = RNG.uniform(31.3, 74.7, n)
    distance = sensor_noise(distance, sigma=2.5)
    distance = np.clip(distance, 10.0, 300.0)

    visibility = clip01(RNG.uniform(0.0, 0.15, n) + RNG.normal(0, 0.02, n))
    visibility = transition_blur(visibility, threshold=VISIBILITY_THRESHOLD)

    player_visible = np.zeros(n)
    has_target     = np.zeros(n)

    return pd.DataFrame({
        "player_visible":           boolean_flip(player_visible),
        "visibility_score":         visibility.round(4),
        "player_distance":          distance.round(4),
        "noise_level":              noise.round(4),
        "paranoia_level":           paranoia.round(4),
        "has_investigation_target": boolean_flip(has_target),
        "action": "alert",
    })


def gen_investigate(n: int) -> pd.DataFrame:
    """
    NPC investigando. Oyó ruido, va hacia la zona sospechosa.
    Rangos base: noise [0.005, 0.58], visibility < 0.50, has_target=1.
    La zona gris con chase vive en visibility ~ 0.4998.
    """
    # Noise: activo — fue lo que disparó este estado
    noise = RNG.uniform(NOISE_THRESHOLD * 2, 0.58, n)
    noise = sensor_noise(noise, sigma=0.02)
    noise = transition_blur(noise, threshold=NOISE_THRESHOLD)
    noise = clip01(noise)

    # Paranoia: variable, puede estar baja si acaba de empezar
    paranoia = RNG.uniform(0.0, 1.0, n)
    paranoia = sensor_noise(paranoia, sigma=0.015)
    paranoia = clip01(paranoia)

    # Distancia: rango real investigate [31.3, 224.0]
    distance = RNG.uniform(31.3, 224.0, n)
    distance = sensor_noise(distance, sigma=4.0)
    distance = np.clip(distance, 10.0, 300.0)

    # Visibility: por debajo del umbral, con solapamiento hacia chase
    visibility = RNG.uniform(0.0, VISIBILITY_THRESHOLD, n)
    visibility = sensor_noise(visibility, sigma=0.02)
    visibility = transition_blur(visibility, threshold=VISIBILITY_THRESHOLD)
    visibility = clip01(visibility)

    # has_target: casi siempre 1, pero sensor puede fallar
    has_target     = np.ones(n)
    player_visible = np.where(visibility > 0.45, 1.0, 0.0)  # lo ve de reojo cerca del umbral

    return pd.DataFrame({
        "player_visible":           boolean_flip(player_visible, prob=0.08),
        "visibility_score":         visibility.round(4),
        "player_distance":          distance.round(4),
        "noise_level":              noise.round(4),
        "paranoia_level":           paranoia.round(4),
        "has_investigation_target": boolean_flip(has_target, prob=0.10),
        "action": "investigate",
    })


def gen_chase(n: int) -> pd.DataFrame:
    """
    NPC persiguiendo. Ve claramente al ladrón.
    Rangos base: visibility [0.50, 0.87], visible=1, has_target=1.
    La zona gris con investigate vive en visibility ~ 0.4998.
    """
    # Visibility: por encima del umbral, con solapamiento hacia investigate
    visibility = RNG.uniform(VISIBILITY_THRESHOLD, 0.87, n)
    visibility = sensor_noise(visibility, sigma=0.025)
    visibility = transition_blur(visibility, threshold=VISIBILITY_THRESHOLD)
    visibility = clip01(visibility)

    # Noise: variable — el ladrón se mueve
    noise = RNG.uniform(0.0, 0.58, n)
    noise = sensor_noise(noise, sigma=0.02)
    noise = clip01(noise)

    # Paranoia: alta en persecución, pero puede empezar baja (chase temprano)
    paranoia = RNG.uniform(0.05, 1.0, n)
    paranoia = sensor_noise(paranoia, sigma=0.015)
    paranoia = clip01(paranoia)

    # Distancia: rango real chase [30.8, 199.9]
    distance = RNG.uniform(30.8, 199.9, n)
    distance = sensor_noise(distance, sigma=3.5)
    distance = np.clip(distance, 10.0, 300.0)

    player_visible = np.ones(n)
    has_target     = np.ones(n)

    return pd.DataFrame({
        "player_visible":           boolean_flip(player_visible, prob=0.06),
        "visibility_score":         visibility.round(4),
        "player_distance":          distance.round(4),
        "noise_level":              noise.round(4),
        "paranoia_level":           paranoia.round(4),
        "has_investigation_target": boolean_flip(has_target, prob=0.06),
        "action": "chase",
    })


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NPC Game Data Simulator")
    print("=" * 55)
    print(f"\n  Umbrales reales (npc_model.json):")
    print(f"    paranoia   threshold : {PARANOIA_THRESHOLD}")
    print(f"    noise      threshold : {NOISE_THRESHOLD}")
    print(f"    visibility threshold : {VISIBILITY_THRESHOLD}")
    print(f"\n  Parámetros de simulación:")
    print(f"    sensor_sigma  : {SENSOR_SIGMA}  (ruido base de sensores)")
    print(f"    blur_zone     : {BLUR_ZONE}  (zona de incertidumbre cerca de umbrales)")
    print(f"    blur_sigma    : {BLUR_SIGMA}  (magnitud de esa incertidumbre)")
    print(f"    flip_prob     : {FLIP_PROB}  (prob. de lectura booleana incorrecta)")
    print(f"\n  Generando {N_PER_STATE} filas por estado...")

    frames = [
        gen_patrol(N_PER_STATE),
        gen_alert(N_PER_STATE),
        gen_investigate(N_PER_STATE),
        gen_chase(N_PER_STATE),
    ]

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\n  Guardado en: {OUT.resolve()}")
    print(f"  Total filas : {len(df)}")
    print(f"\n  Distribución:")
    for action, count in df["action"].value_counts().items():
        print(f"    {action:<12} {count}")

    # Verificar solapamiento introducido
    print(f"\n  Solapamiento en zonas de transición:")
    patrol_over  = (df[df.action=="patrol"]["paranoia_level"] > PARANOIA_THRESHOLD).sum()
    alert_under  = (df[df.action=="alert"]["paranoia_level"]  < PARANOIA_THRESHOLD).sum()
    inv_over_vis = (df[df.action=="investigate"]["visibility_score"] > VISIBILITY_THRESHOLD).sum()
    chase_under  = (df[df.action=="chase"]["visibility_score"] < VISIBILITY_THRESHOLD).sum()
    print(f"    patrol  con paranoia > {PARANOIA_THRESHOLD}: {patrol_over} filas")
    print(f"    alert   con paranoia < {PARANOIA_THRESHOLD}: {alert_under} filas")
    print(f"    invest. con visibility > {VISIBILITY_THRESHOLD}: {inv_over_vis} filas")
    print(f"    chase   con visibility < {VISIBILITY_THRESHOLD}: {chase_under} filas")
    print(f"\n  Listo. Corre 'python train.py' para entrenar con estos datos.\n")


if __name__ == "__main__":
    main()