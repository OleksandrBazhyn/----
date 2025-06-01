import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def get_points_from_user():
    N = int(input("Введіть кількість точок (N, до 100): "))
    points = []
    for i in range(N):
        while True:
            try:
                x, y = map(float, input(f"Введіть координати {i+1}-ї точки (x y): ").split())
                points.append([x, y])
                break
            except ValueError:
                print("Помилка вводу! Введіть дві координати через пробіл, наприклад: 2.5 4.1")
    return np.array(points)

def get_points_random():
    N = int(input("Введіть кількість випадкових точок (N, до 1000): "))
    size = float(input("Введіть розмір області (наприклад, 10): "))
    points = np.random.rand(N, 2) * size
    print(f"Згенеровано {N} точок у квадраті розміром {size}x{size}")
    return points

def point_to_line_distance(a, b, p):
    return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

def min_distance_to_polygon(polygon, point):
    n = len(polygon)
    min_dist = np.inf
    for i in range(n):
        a = polygon[i]
        b = polygon[(i+1)%n]
        d = point_to_line_distance(a, b, point)
        min_dist = min(min_dist, d)
    return min_dist

def is_point_in_polygon(polygon, point):
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
    centroid = np.mean(polygon, axis=0)
    def objective(point):
        if not is_point_in_polygon(polygon, point):
            return 1e6  # велике число (за межами полігону)
        return -min_distance_to_polygon(polygon, point)
    res = minimize(objective, centroid, method='Nelder-Mead')
    center = res.x
    radius = min_distance_to_polygon(polygon, center)
    return center, radius

def plot_polygon_with_circle(polygon, center, radius):
    plt.figure(figsize=(12,9))
    pts = np.vstack([polygon, polygon[0]])
    plt.plot(pts[:,0], pts[:,1], 'k-', lw=2, label='Опукла оболонка')
    plt.plot(center[0], center[1], 'ro', label='Центр кола')
    circle = plt.Circle(center, radius, color='r', fill=False, lw=2, label='Вписане коло')
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.legend()
    plt.title('Найбільше вписане коло в опуклій оболонці')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def main():
    print("==== Програма пошуку найбільшого вписаного кола в опуклій оболонці ====")
    print("Оберіть режим:")
    print("1 — Ручний ввід координат точок")
    print("2 — Автоматична генерація випадкових точок")
    while True:
        mode = input("Ваш вибір (1 або 2): ").strip()
        if mode in ('1', '2'):
            break
        print("Некоректний вибір. Спробуйте ще раз.")
    if mode == '1':
        points = get_points_from_user()
    else:
        points = get_points_random()

    # Побудова опуклої оболонки
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    
    print(f"\nКількість вершин опуклої оболонки: {len(polygon)}")
    print("Координати вершин опуклої оболонки (у порядку обходу):")
    for i, (x, y) in enumerate(polygon):
        print(f"{i+1}: ({x:.4f}, {y:.4f})")

    # Пошук кола
    center, radius = find_largest_inscribed_circle(polygon)
    print(f"\nКоординати центру кола: ({center[0]:.4f}, {center[1]:.4f})")
    print(f"Радіус вписаного кола: {radius:.4f}")
    
    # Графіка
    plot_polygon_with_circle(polygon, center, radius)

if __name__ == "__main__":
    main()
