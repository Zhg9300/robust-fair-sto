"""Server-side aggregation rules for flattened client gradients."""

from components.aggregator.base import Mean
from components.aggregator.robust import (
    CenteredClipping,
    CoordinateWiseMedian,
    CoordinateWiseTrimmedMean,
    FABA,
    GeometricMedian,
    NormBasedScreening,
)


def build_aggregator(name="mean", **options):
    builders = {
        "mean": Mean,
        "cwtm": CoordinateWiseTrimmedMean,
        "cwm": CoordinateWiseMedian,
        "median": lambda: GeometricMedian(
            max_iterations=options.get("median_max_iterations", 100),
            tolerance=options.get("aggregator_tolerance", 1e-6),
        ),
        "faba": FABA,
        "centered_clipping": lambda: CenteredClipping(
            clipping_radius=options.get("cc_tau", 10.0),
            iterations=options.get("cc_iterations", 1),
        ),
        "nbs": NormBasedScreening,
    }
    try:
        return builders[name]()
    except KeyError as exc:
        supported = ", ".join(builders)
        raise ValueError(
            f"Unknown aggregator {name!r}. Supported aggregators: {supported}."
        ) from exc
