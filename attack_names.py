"""Canonical attack names shared by training and result readers."""


def normalize_attack_mode(value):
    """Accept legacy identifiers while using mimic_attack for new runs."""
    if isinstance(value, str) and value.strip().lower() in {
        "adaptive_copying", "adaptive_copying_attack", "mimic_attack",
    }:
        return "mimic_attack"
    return value
