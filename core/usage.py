
import time

USAGE = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "start_time": None,
}


def reset_usage():
    USAGE["calls"] = 0
    USAGE["input_tokens"] = 0
    USAGE["output_tokens"] = 0
    USAGE["start_time"] = time.perf_counter()


def record_usage(response):
    USAGE["calls"] += 1

    if response.usage:
        USAGE["input_tokens"] += response.usage.input_tokens
        USAGE["output_tokens"] += response.usage.output_tokens


def get_usage():
    return {
        "calls": USAGE["calls"],
        "input_tokens": USAGE["input_tokens"],
        "output_tokens": USAGE["output_tokens"],
        "elapsed_seconds": round(
            time.perf_counter() - USAGE["start_time"], 2
        ),
    }
