# Vector Bench Dist

> Empirical execution time profiling and statistical dispersion analysis comparing Python loop iterations against vectorized SIMD processing via NumPy.

## Overview
High-throughput mathematical workloads written in native Python often suffer from significant runtime overhead due to object boxing, dynamic type lookups, and pointer-chasing during loop iterations. Standard Python constructs like list comprehensions and generator expressions incur notable call stack pressure when scaling to hundreds of thousands of multidimensional spatial coordinate evaluations.

`vector-bench-dist` provides a reproducible runtime benchmark that contrasts pure Python standard library routines (`math.sqrt`, generator sum expressions) with contiguous memory C-array vectorized computations implemented in NumPy.

## How It Works
The benchmark pipeline executes the following sequence:

1. **Dataset Generation:** Synthesizes high-density multi-dimensional floating-point coordinate arrays ($500,000 \times 3$) using uniform distribution sampling via `np.random.uniform`.
2. **Data Structural Unpacking:** Converts NumPy C-contiguous memory blocks into nested native Python lists via `.tolist()` to ensure a fair execution baseline devoid of memory conversion overhead during measurement.
3. **Iterative Benchmark Processing:**
   - Evaluates Euclidean distance across matching pairs using nested generator expressions inside `math.sqrt(sum((val_a - val_b) ** 2 ...))`.
   - Computes sample standard deviation ($N-1$ degrees of freedom) by iterating over calculated distance arrays.
4. **Vectorized Benchmark Processing:**
   - Computes differences across spatial axes using NumPy broadcasting: `np.sqrt(np.sum((coords_a - coords_b) ** 2, axis=1))`.
   - Evaluates sample standard deviation using SIMD execution pathways (`np.std(..., ddof=1)`).
5. **Telemetry & Profiling:** Measures high-resolution process time using `time.perf_counter()` to yield relative execution speedup metrics.

## Key Features
- **High-Precision Timing:** Uses monotonicity-guaranteed `time.perf_counter()` clocks to capture raw execution intervals down to sub-millisecond precision.
- **Statistical Parity:** Standard deviation implementations across both native Python and NumPy explicitly lock Bessel's correction (`ddof=1` sample variance) to guarantee mathematical parity.
- **N-Dimensional Support:** Configurable array shapes allow stress-testing point clouds across arbitrary spatial dimensions.

## Tech Stack & Core Dependencies Breakdown
- **Python Target:** Python 3.10+ (utilizes modern type hint syntax such as `list[list[float]]`).
- **Standard Library:**
  - `math`: Used for `math.sqrt()` standard scalar calculations.
  - `time`: Used for `time.perf_counter()` high-resolution monotonic timing execution checks.
- **Third-Party Packages:**
  - `numpy`: Provides strided memory arrays and vectorized operations.

## Environment & Web-Based Quick Start

### Running in GitHub Codespaces
1. Open this repository in GitHub by clicking **Code** > **Codespaces** > **Create codespace on main**.
2. Once the web container boots, execute:
   ```bash
   python main.py
   ```

### Local Virtual Environment Setup (venv)
```bash
# Clone and enter directory
git clone [https://github.com/your-username/vector-bench-dist.git](https://github.com/your-username/vector-bench-dist.git)
cd vector-bench-dist

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy

# Execute the benchmark harness
python main.py
```

### Dependency Management with UV
```bash
uv venv
source .venv/bin/activate
uv pip install numpy
python main.py
```

## Repository Structure

```bash
vector-bench-dist/
├── .github/
│   └── workflows/
│       └── ci.yml          # Automated linting and test execution workflow
├── .gitignore              # Stack-specific rule set ignoring byte-code and virtual environments
├── LICENSE                 # MIT Open-Source License
├── README.md               # Architecture specification and execution manual
└── main.py                 # Core benchmark pipeline and data generation entry point
```

## Roadmap

**[ ] Memory Footprint Tracking:** Integrate tracemalloc to record peak heap allocation across native list generation vs strided array creation.

**[ ] Multi-Core Parallelization:** Add execution paths comparing Python's concurrent.futures.ProcessPoolExecutor against NumPy's underlying OpenMP thread pool execution.

**[ ] Automated CLI Instrumentation:** Migrate hardcoded hyper-parameters (num_points, dimensions) to a argparse CLI interface.