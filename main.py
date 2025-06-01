import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def point_to_line_distance(a, b, p):
    # Відстань від точки p до прямої через a, b
    return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

def random_convex_polygon(N=8, size=10):
    points = np.random.rand(N, 2) * size
    hull = ConvexHull(points)
    return points[hull.vertices]

def min_distance_to_polygon(polygon, point):
    # Мінімальна відстань до будь-якої сторони полігона
    n = len(polygon)
    min_dist = np.inf
    for i in range(n):
        a = polygon[i]
        b = polygon[(i+1)%n]
        d = point_to_line_distance(a, b, point)
        min_dist = min(min_dist, d)
    return min_dist

def is_point_in_polygon(polygon, point):
    # Для опуклого полігону: всі знаки скалярних добутків однакові
    n = len(polygon)
    sign = None
    for i in range(n):
        a = polygon[i]
        b = polygon[(i+1)%n]
        edge = b - a
        to_p = point - a
        cross = np.cross(edge, to_p)
        if cross == 0:
            continue  # на ребрі
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    return True

def find_largest_inscribed_circle(polygon):
    # Початкове припущення: центр мас полігону
    centroid = np.mean(polygon, axis=0)
    def objective(point):
        # Мінус мінімальна відстань до сторін (бо minimize)
        if not is_point_in_polygon(polygon, point):
            return 1e6  # велике число (за межами полігону)
        return -min_distance_to_polygon(polygon, point)
    res = minimize(objective, centroid, method='Nelder-Mead')
    center = res.x
    radius = min_distance_to_polygon(polygon, center)
    return center, radius

def plot_polygon_with_circle(polygon, center, radius):
    plt.figure(figsize=(7,7))
    pts = np.vstack([polygon, polygon[0]])
    plt.plot(pts[:,0], pts[:,1], 'k-', lw=2, label='Полігон')
    plt.plot(center[0], center[1], 'ro', label='Центр кола')
    circle = plt.Circle(center, radius, color='r', fill=False, lw=2, label='Вписане коло')
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.legend()
    plt.title('Вписане коло найбільшого радіуса')
    plt.show()

# ==== ТЕСТ ====
N = np.random.randint(6, 13)
polygon = random_convex_polygon(N=N, size=10)

hull = ConvexHull(polygon)
polygon = polygon[hull.vertices]

center, radius = find_largest_inscribed_circle(polygon)
plot_polygon_with_circle(polygon, center, radius)
print(f"Кількість вершин: {len(polygon)}")
print(f"Центр: {center}")
print(f"Радіус: {radius}")
