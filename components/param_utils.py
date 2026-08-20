import ast
import copy


FULL_BATCH = "full"


def parse_batch_size(value):
    """Parse a positive integer batch size or the explicit ``full`` mode."""
    if isinstance(value, str) and value.strip().lower() == FULL_BATCH:
        return FULL_BATCH
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("B must be a positive integer or 'full'.") from exc
    if batch_size <= 0:
        raise ValueError("B must be a positive integer or 'full'.")
    return batch_size


def parse_min_client_samples(value):
    """Parse the pathological-partition minimum or its automatic policy."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        minimum = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_client_samples must be 'auto' or a non-negative integer.") from exc
    if minimum < 0:
        raise ValueError("min_client_samples must be 'auto' or a non-negative integer.")
    return minimum


def parse_bool(value, name="value"):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{name} must be a boolean value, got {value!r}.")

    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off", "none", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


def parse_id_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() == "none":
            return []
        if text.startswith("["):
            value = ast.literal_eval(text)
        else:
            value = [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def build_optimizer_like(template, parameters, lr=None):
    options = copy.deepcopy(template.defaults)
    if lr is not None:
        options["lr"] = lr
    return template.__class__(parameters, **options)
