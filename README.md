# NPC AI — Videojuego de Sigilo

Sistema de inteligencia artificial para un NPC en un videojuego de sigilo desarrollado en Godot. El jugador es un ladrón que debe robar objetos de una casa mientras evita al dueño. La IA del dueño toma decisiones en tiempo real combinando un modelo de Machine Learning (Random Forest) con pathfinding A*.

---

## Descripción del Proyecto

El dueño de la casa (NPC) tiene cuatro estados de comportamiento:

| Estado | Descripción |
|---|---|
| `patrol` | Recorre la casa normalmente sin estímulos |
| `alert` | Paranoia acumulada, sin estímulo activo |
| `investigate` | Oyó ruido, busca la fuente |
| `chase` | Ve al ladrón directamente, lo persigue |

El modelo predice el estado en cada tick del juego basándose en seis features capturadas en tiempo real. El pathfinding A* calcula la ruta óptima hacia el objetivo correspondiente a ese estado.

---

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd IA_ProyectoFinal

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### 1. Explorar el dataset original
```bash
python eda.py
```

### 2. Generar dataset simulado
```bash
python data/game_data_simulator.py
```

### 3. Entrenar el modelo
```bash
python train.py
```
Ejecuta el pipeline completo: carga, limpieza, entrenamiento, exportación y reporte. Genera `models/npc_model.json` y `models/training_report.png`.

### 4. Ejecutar el minijuego interactivo
```bash
python -m visualization.astar_plot
```

| Control | Acción |
|---|---|
| Flechas | Mover al ladrón |
| `E` | Esconderse (junto a una caja) |
| `Q` | Salir |

---

### Modelo

Random Forest con los siguientes hiperparámetros:

```python
n_estimators     = 100
max_depth        = 5
min_samples_leaf = 10
min_samples_split= 20
class_weight     = "balanced"
```

### Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 0.978 |
| Macro F1 | 0.978 |
| Macro Precision | 0.978 |
| Macro Recall | 0.978 |


### Exportación a Godot

El modelo se exporta como JSON con los 100 árboles serializados, permitiendo inferencia directa en Godot sin dependencias externas de Python.

---

## Pathfinding A*

Reimplementación del algoritmo A* de Godot en Python, garantizando comportamiento idéntico en ambos entornos. Utiliza heurística Manhattan para grids de 4 direcciones.

El NPC recalcula su path en cada tick hacia:
- **Waypoints de patrulla** cuando no tiene objetivo activo
- **Última posición conocida del ladrón** cuando tiene objetivo

Cuando el ladrón se esconde, el NPC pierde el objetivo inmediatamente, investiga la última posición conocida, y si no encuentra nada retoma la patrulla normal.

---