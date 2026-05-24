from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Vector2i:
    x: int
    y: int

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        return Vector2i(self.x + other.x, self.y + other.y)

# Direcciones equivalentes a Vector2i.UP/DOWN/LEFT/RIGHT en Godot
UP    = Vector2i(0, -1)
DOWN  = Vector2i(0,  1)
LEFT  = Vector2i(-1, 0)
RIGHT = Vector2i(1,  0)


class AStarNode:
    def __init__(self, position: Vector2i):
        self.position = position
        self.g_cost: int = float('inf')
        self.h_cost: int = 0
        self.parent: Optional['AStarNode'] = None

    def get_f_cost(self) -> int:
        return self.g_cost + self.h_cost


def heuristic(a: Vector2i, b: Vector2i) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)

def get_neighbors(pos: Vector2i) -> list[Vector2i]:
    return [pos + UP, pos + DOWN, pos + LEFT, pos + RIGHT]

def get_lowest_f_cost_node(nodes: list[AStarNode]) -> AStarNode:
    best_node = nodes[0]
    for node in nodes:
        if node.get_f_cost() < best_node.get_f_cost():
            best_node = node
        elif node.get_f_cost() == best_node.get_f_cost():
            if node.h_cost < best_node.h_cost:
                best_node = node
    return best_node

def reconstruct_path(end_node: AStarNode) -> list[Vector2i]:
    path = []
    current = end_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    path.reverse()
    return path

def find_node_in_open(nodes: list[AStarNode], position: Vector2i) -> Optional[AStarNode]:
    for node in nodes:
        if node.position == position:
            return node
    return None

def find_path(start: Vector2i, goal: Vector2i, grid_manager) -> list[Vector2i]:
    open_list: list[AStarNode] = []
    closed_set: dict = {}

    start_node = AStarNode(start)
    start_node.g_cost = 0
    start_node.h_cost = heuristic(start, goal)
    open_list.append(start_node)

    while len(open_list) > 0:
        current = get_lowest_f_cost_node(open_list)

        if current.position == goal:
            return reconstruct_path(current)

        open_list.remove(current)
        closed_set[current.position] = True

        for neighbor_pos in get_neighbors(current.position):
            if not grid_manager.is_walkable(neighbor_pos):
                continue
            if neighbor_pos in closed_set:
                continue

            tentative_g_cost = current.g_cost + 1
            neighbor = find_node_in_open(open_list, neighbor_pos)

            if neighbor is None:
                neighbor = AStarNode(neighbor_pos)
                open_list.append(neighbor)

            if tentative_g_cost < neighbor.g_cost:
                neighbor.g_cost = tentative_g_cost
                neighbor.h_cost = heuristic(neighbor_pos, goal)
                neighbor.parent = current

    return []