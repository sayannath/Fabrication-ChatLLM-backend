import logging
import sys
import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure stdlib logging + structlog to emit JSON logs.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # adds contextvars (e.g., request_id)
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),  # <-- JSON output
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                timestamper,
            ],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Make uvicorn loggers less noisy but still structured
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [handler]
        logging.getLogger(name).setLevel(
            getattr(logging, log_level.upper(), logging.INFO)
        )
