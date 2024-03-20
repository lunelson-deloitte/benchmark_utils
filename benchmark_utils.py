import timeit
import functools

import pandas as pd
import seaborn as sns

sns.set_theme(rc={"figure.figsize": (25, 15)})


def benchmark_workflows(
    workflows: dict[str, callable],
    _globals: dict[str, str],
    n_repeat: int = 5,
    n_number: int = 3,
    range_start: int = 20,
    range_end: int = 100 + 1,
    range_step: int = 20,
) -> dict[str, float]:
    """Benchmark multiple workflows"""
    benchmark_func = functools.partial(
        timeit.repeat, repeat=n_repeat, number=n_number, globals=_globals
    )
    benchmark_results = {
        workflow_name: {
            f"{frac}%": benchmark_func(
                f'{workflow_func}(readin_func="cp.read", input_path="path/to/file.parquet", fraction={float(frac / 100):.0f})'
            )
            for frac in range(range_start, range_end, range_step)
        }
        for workflow_name, workflow_func in workflows.items()
    }
    return benchmark_results


def benchmark_operations(
    operations: list[str],
    workflows: dict[str, callable],
    _globals: dict[str, str],
    n_repeat: int = 5,
    n_number: int = 3,
    range_start: int = 20,
    range_end: int = 100 + 1,
    range_step: int = 20,
) -> dict[str, float]:
    """Benchmark multiple workflows disaggregated by operation"""
    benchmark_results = {
        operation: benchmark_workflows(
            workflows,
            _globals,
            n_repeat=n_repeat,
            n_number=n_number,
            range_start=range_start,
            range_end=range_end,
            range_step=range_step,
        )
        for operation in operations
    }
    return benchmark_results


def summarize_benchmark(
    benchmark_results: dict[str, list[float]], operations_included: bool = False
) -> pd.DataFrame:
    """Convert benchmark results into pandas DataFrame"""
    data = pd.DataFrame.from_dict(benchmark_results, orient="index")

    if not operations_included:
        return (
            data.reset_index()
            .rename(columns={"index": "Workflow"})
            .melt(id_vars="Workflow", var_name="Sample Percentage", value_name="Time")
            .explode("Time")
            .reset_index(drop=True)
        )

    data = (
        data.reset_index()
        .rename(columns={"index": "Operation"})
        .melt(id_vars="Operation", var_name="Workflow", value_name="Benchmarks")
    )

    data_full = (
        pd.concat(
            [data.drop(columns="Benchmarks"), data["Benchmarks"].apply(pd.Series)],
            axis=1,
        )
        .melt(
            id_vars=["Operation", "Workflow"],
            var_name="Sample Percentage",
            value_name="Time",
        )
        .explode("Time")
        .reset_index(drop=True)
    )

    return data_full


def save_benchmark(
    data: pd.DataFrame, output_path: str, write_func: callable, overwrite: bool = False
) -> None:
    """Write benchmark results (pandas DataFrame) to path"""
    if overwrite:
        write_func(data, output_path, "overwrite", register=True)
