import numpy as np
import matplotlib.pyplot as plt

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm else v

def line_from_points(p1, p2):
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = -(A * p1[0] + B * p1[1])
    return (A, B, C)

def bisector(edge1, edge2):
    (p1, p2) = edge1
    (q1, q2) = edge2
    n1 = normalize(np.array([p2[1] - p1[1], p1[0] - p2[0]]))
    n2 = normalize(np.array([q2[1] - q1[1], q1[0] - q2[0]]))
    bis = normalize(n1 + n2)
    if np.allclose(p2, q1):
        base_point = p2
    elif np.allclose(p1, q2):
        base_point = p1
    else:
        base_point = (p2 + q1) / 2
    A = -bis[1]
    B = bis[0]
    C = -(A * base_point[0] + B * base_point[1])
    return (A, B, C)

def intersection(L1, L2):
    A1, B1, C1 = L1
    A2, B2, C2 = L2
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-12:
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det
    return np.array([x, y])

def point_line_distance(p, line):
    A, B, C = line
    return abs(A * p[0] + B * p[1] + C) / np.sqrt(A ** 2 + B ** 2)

def incenter_of_triangle(triangle):
    a, b, c = triangle
    la = np.linalg.norm(b - c)
    lb = np.linalg.norm(a - c)
    lc = np.linalg.norm(a - b)
    P = la + lb + lc
    center = (la * a + lb * b + lc * c) / P
    s = P / 2
    area = abs(
        (a[0] * (b[1] - c[1]) +
         b[0] * (c[1] - a[1]) +
         c[0] * (a[1] - b[1])) / 2.0)
    radius = area / s
    return center, radius

def plot_polygon_and_circle(edges, center=None, radius=None, remove_idx=None, step=None):
    plt.figure(figsize=(7,7))
    n = len(edges)
    polygon_points = [edge[0] for edge in edges] + [edges[0][0]]
    polygon_points = np.array(polygon_points)
    plt.plot(polygon_points[:,0], polygon_points[:,1], 'k-o', label="Polygon")
    plt.fill(polygon_points[:,0], polygon_points[:,1], alpha=0.07)
    if center is not None and radius is not None:
        circle = plt.Circle(center, radius, color='b', fill=False, linewidth=2, label="Current circle")
        plt.gca().add_patch(circle)
        plt.plot(center[0], center[1], 'bo')
    if remove_idx is not None:
        e = edges[remove_idx]
        ex = [e[0][0], e[1][0]]
        ey = [e[0][1], e[1][1]]
        plt.plot(ex, ey, 'r-', linewidth=4, label="Edge to remove")
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Step {step}" if step is not None else "")
    plt.legend()
    plt.show()

def compressing_sides_algorithm_with_plot(polygon):
    n = len(polygon)
    verts = [np.array(p) for p in polygon]
    edges = [(verts[i], verts[(i + 1) % n]) for i in range(n)]
    bisectors = []
    for i in range(n):
        prev = edges[i - 1]
        curr = edges[i]
        bisectors.append(bisector(prev, curr))
    step = 1
    while len(edges) > 3:
        heights = []
        centers = []
        for i, edge in enumerate(edges):
            left_bis = bisectors[i]
            right_bis = bisectors[(i + 1) % len(edges)]
            pt = intersection(left_bis, right_bis)
            centers.append(pt)
            if pt is not None:
                height = point_line_distance(pt, line_from_points(*edge))
            else:
                height = float('inf')
            heights.append(height)
        min_idx = np.argmin(heights)
        # Малюємо поточний стан
        plot_polygon_and_circle(
            edges,
            center=centers[min_idx],
            radius=heights[min_idx],
            remove_idx=min_idx,
            step=step
        )
        step += 1
        # Видаляємо ребро з мінімальною висотою
        edges.pop(min_idx)
        bisectors.pop(min_idx)
        # Оновлюємо суміжні бісектриси
        prev_idx = (min_idx - 1) % len(edges)
        curr_idx = min_idx % len(edges)
        bisectors[curr_idx] = bisector(edges[prev_idx], edges[curr_idx])
    # Для трикутника
    triangle = [edges[0][0], edges[1][0], edges[2][0]]
    center, radius = incenter_of_triangle(triangle)
    plot_polygon_and_circle(edges, center=center, radius=radius, step=step)
    return center, radius

# === ПРИКЛАД ВИКОРИСТАННЯ ===
polygon = [
    (0, 0),
    (4, 0),
    (5, 3),
    (2, 5),
    (-1, 3)
]
center, radius = compressing_sides_algorithm_with_plot(polygon)
print("Center:", center)
print("Radius:", radius)
