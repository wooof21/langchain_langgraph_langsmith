
import logging
import json
import time

from datetime import datetime, timezone
from functools import wraps
from typing import Callable

from langchain_openai import ChatOpenAI
from langsmith import traceable


# =========================================================
# Logging
# =========================================================

class JSONFormatter(logging.Formatter):

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging():

    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


logger = setup_logging()


# =========================================================
# Metrics
# =========================================================

class MetricsCollector:

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
    ):
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

    def get_summary(self):

        requests = self.metrics["requests_total"]

        avg_latency = (
            self.metrics["latency_sum"] /
            self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )

        error_rate = (
            self.metrics["errors_total"] / requests
            if requests > 0
            else 0
        )

        return {
            "total_requests": requests,
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
        }


metrics = MetricsCollector()

def monitored(func: Callable):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            metrics.record_request(
                latency_ms=latency_ms,
                error=False,
            )

            logger.info(
                "Function completed",
                extra={
                    "extra_data": {
                        "function": func.__name__,
                        "latency_ms": round(latency_ms, 2),
                        "success": True,
                    }
                },
            )

            return result

        except Exception as e:

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            metrics.record_request(
                latency_ms=latency_ms,
                error=True,
            )

            logger.exception(
                "Function failed",
                extra={
                    "extra_data": {
                        "function": func.__name__,
                        "latency_ms": round(latency_ms, 2),
                        "error": str(e),
                        "success": False,
                    }
                },
            )

            raise

    return wrapper


#=================================
# Use
#=================================

class InstrumentedLLM:

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

    @traceable(name="instrumented_invoke")
    @monitored
    def invoke(self, query: str) -> str:

        response = self.llm.invoke(query)

        return response.content


"""
Limitation: token metrics

A generic decorator can easily measure:
    latency
    success/failure
    exception
    function name
    request count
    error count

But token usage is LLM-specific.

The decorator doesn't necessarily know that:
    `result`
contains an AI message with token metadata.


For LLM decorator, nned to specifically inspect the response.

For example, LangChain AI messages can expose usage metadata, depending on the model/provider:
    response = self.llm.invoke(query)

    print(response.usage_metadata)


Separate normal function decorator and LLM decorator:
                    Observability

          ┌──────────────┴──────────────┐
          │                             │
     @monitored                    @monitor_llm
          │                             │
     Any function                    LLM calls
          │                             │
    ┌─────┴─────┐              ┌────────┴────────┐
    │           │              │                 │
 latency      errors        latency            errors
 requests     logging       tokens             logging
                            model usage

*****: decorator order
    -> The top decorator is the outermost wrapper.
    
It determines whether record one metric for the whole logical request 
or one metric for every retry attempt

One metric for the whole logical request -> Put @monitored outside @retry: @monitored, @retry
@monitored
    │
    │ start timer
    ▼
@retry
    │
    ├── attempt #1 ❌
    ├── attempt #2 ❌
    └── attempt #3 ✅
    │
    ▼
@monitored
    │ stop timer
    │ record metric
    ▼
  return

Latency includes everything:
        attempt #1
        + backoff
        + attempt #2
        + backoff
        + attempt #3
        ----------------
        total latency
e.g.
        attempt 1    500ms  ❌
        wait        1000ms
        attempt 2    500ms  ❌
        wait        2000ms
        attempt 3    500ms  ✅
        -----------------------
        metric       4500ms

One metric for every retry attempt -> @retry, @monitored

        @retry
        │
        ├── @monitored
        │       │
        │     attempt #1 ❌
        │       │
        │     metric #1
        │
        ├── @monitored
        │       │
        │     attempt #2 ❌
        │       │
        │     metric #2
        │
        └── @monitored
                │
                attempt #3 ✅
                │
                metric #3

metric would show:
    requests_total = 3
    errors_total   = 2


In production, may want both:
For production LangChain project, that separation—request-level metrics vs 
attempt-level metrics—is better than trying to make one 
@monitored decorator represent both.
"""

# Make an LLM-specific decorator:

def monitor_llm(func: Callable):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        try:
            response = func(*args, **kwargs)

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            usage = response.usage_metadata or {}

            input_tokens = usage.get(
                "input_tokens",
                0
            )

            output_tokens = usage.get(
                "output_tokens",
                0
            )

            metrics.record_request(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            logger.info(
                "LLM request completed",
                extra={
                    "extra_data": {
                        "function": func.__name__,
                        "latency_ms": round(latency_ms, 2),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
            )

            return response

        except Exception as e:

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            metrics.record_request(
                latency_ms=latency_ms,
                error=True,
            )

            logger.exception(
                "LLM request failed",
                extra={
                    "extra_data": {
                        "function": func.__name__,
                        "latency_ms": round(latency_ms, 2),
                        "error": str(e),
                    }
                },
            )

            raise

    return wrapper

# Then change the method slightly so the decorated function returns 
# the actual LangChain AIMessage

class InstrumentedLLM:

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

    @traceable(name="instrumented_invoke")
    @monitor_llm
    def invoke_raw(self, query: str):
        return self.llm.invoke(query)

    def invoke(self, query: str) -> str:
        response = self.invoke_raw(query)
        return response.content