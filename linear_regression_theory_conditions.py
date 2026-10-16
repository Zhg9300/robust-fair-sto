"""Table 2 certificates and Appendix B bounds for X=I_20, h=8, b=2.

This module has no training dependencies so command previews can validate the
theory conditions before importing PyTorch or creating result files.
"""

import math


CERTIFICATE_FIELDS = (
    "theory_rho", "theory_R", "theory_separation_rhs", "theory_separation_margin",
)


def theory_certificate(aggregator, noise_std, delta):
    """Return the finite certificate, or empty fields for comparison baselines."""
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be positive and finite")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    n, b, h, d = 10, 2, 8, 20
    diameter_factor = min(2 * math.sqrt(n - b), math.sqrt(d))
    certificates = {
        "cwm": n / (2 * (n - b)) * diameter_factor,
        "cwtm": b / (n - b) * diameter_factor,
        "median": 1 + (n - b) / math.sqrt(n * (n - 2 * b)),
        "krum": 1 + math.sqrt((n - b) / (n - 2 * b)),
        "mda": 2 * b / (n - b),
        "faba": 2 * b / (n - 3 * b),
    }
    if aggregator in {"mean", "nbs"}:
        return dict.fromkeys(CERTIFICATE_FIELDS, "")
    if aggregator not in certificates:
        raise ValueError(f"No Table 2 certificate for aggregator {aggregator!r}")
    rho = certificates[aggregator]
    radius = noise_std * math.sqrt(d) * (
        1 / math.sqrt(h - 1)
        + 2 * (rho + 1 / h) * (1 + math.sqrt(2 * math.log(h - 1)))
    )
    rhs = h * radius
    if not math.isfinite(rhs):
        raise ValueError("The theory separation bound must be finite")
    return dict(zip(CERTIFICATE_FIELDS, (rho, radius, rhs, delta - rhs)))


def validate_separation(aggregators, noise_std, delta):
    """Require the strict separation condition for each certified aggregator."""
    for aggregator in aggregators:
        certificate = theory_certificate(aggregator, noise_std, delta)
        if (certificate["theory_separation_margin"] != ""
                and certificate["theory_separation_margin"] <= 0):
            raise ValueError(
                f"{aggregator}: theory requires delta > h*R; "
                f"delta={delta:g}, h*R={certificate['theory_separation_rhs']:.9g}. "
                "Decrease --noise-std or increase --delta."
            )
