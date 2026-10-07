import math
import time
import numpy as np


def pure_python_euclidean(
    coords_a: list[list[float]], coords_b: list[list[float]]
) -> list[float]:
    """Calculates Euclidean distance between coordinate pairs using pure Python loops."""
    distances = []
    for point_a, point_b in zip(coords_a, coords_b):
        # Calculate squared differences across dimensions
        squared_diff_sum = sum(
            (val_a - val_b) ** 2 for val_a, val_b in zip(point_a, point_b)
        )
        distances.append(math.sqrt(squared_diff_sum))
    return distances


def pure_python_std(data: list[float]) -> float:
    """Calculates standard deviation of a 1D list using pure Python math."""
    n = len(data)
    if n < 2:
        return 0.0

    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)  # Sample std dev
    return math.sqrt(variance)


def numpy_vectorized_processing(
    coords_a: np.ndarray, coords_b: np.ndarray
) -> tuple[np.ndarray, float]:
    """Calculates Euclidean distances and standard deviation using vectorized NumPy ops."""
    # Vectorized Euclidean Distance across axis 1 (dimensions)
    distances = np.sqrt(np.sum((coords_a - coords_b) ** 2, axis=1))
    # Vectorized Sample Standard Deviation (ddof=1 matches sample std dev)
    std_dev = np.std(distances, ddof=1)
    return distances, std_dev


def main():
    # Parameters for the dataset
    num_points = 500_000
    dimensions = 3

    print(
        f"Generating dataset: {num_points:,} pairs of {dimensions}D coordinates..."
    )

    # 1. Prepare NumPy Arrays
    np_a = np.random.uniform(0.0, 100.0, size=(num_points, dimensions))
    np_b = np.random.uniform(0.0, 100.0, size=(num_points, dimensions))

    # 2. Convert to native Python lists for fair comparison
    list_a = np_a.tolist()
    list_b = np_b.tolist()

    print("\n--- Running Pure Python Iterative Benchmark ---")
    start_time = time.perf_counter()
    py_distances = pure_python_euclidean(list_a, list_b)
    py_std = pure_python_std(py_distances)
    py_elapsed = time.perf_counter() - start_time
    print(f"Pure Python Execution Time: {py_elapsed:.4f} seconds")
    print(f"Standard Deviation: {py_std:.4f}")

    print("\n--- Running NumPy Vectorized Benchmark ---")
    start_time = time.perf_counter()
    np_distances, np_std = numpy_vectorized_processing(np_a, np_b)
    np_elapsed = time.perf_counter() - start_time
    print(f"NumPy Execution Time:       {np_elapsed:.4f} seconds")
    print(f"Standard Deviation: {np_std:.4f}")

    # 3. Performance Summary
    speedup = py_elapsed / np_elapsed if np_elapsed > 0 else 0
    print("\n--- Summary ---")
    print(f"NumPy is ~{speedup:.1f}x faster than standard Python loops.")


if __name__ == "__main__":
    main()