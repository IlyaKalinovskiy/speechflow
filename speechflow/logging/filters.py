import logging

__all__ = ["set_logging_filters"]


class AssertionErrorFilter(logging.Filter):
    def filter(self, record):
        return "AssertionError" not in record.getMessage()


class ValueErrorFilter(logging.Filter):
    def filter(self, record):
        return "ValueError" not in record.getMessage()


class IndexErrorFilter(logging.Filter):
    def filter(self, record):
        return "IndexError" not in record.getMessage()


class TimestampErrorFilter(logging.Filter):
    def filter(self, record):
        return "timestamps.py" not in record.getMessage()


class TextProcessorErrorFilter(logging.Filter):
    def filter(self, record):
        return "ZeroSilTokensError" not in record.getMessage()


class LPCProcessorErrorFilter(logging.Filter):
    def filter(self, record):
        return "LPCError" not in record.getMessage()


class CollatedErrorFilter(logging.Filter):
    def filter(self, record):
        return "collate" not in record.getMessage()


def set_logging_filters(logger, **kwargs):
    logger.addFilter(AssertionErrorFilter())
    logger.addFilter(ValueErrorFilter())
    logger.addFilter(IndexErrorFilter())
    logger.addFilter(TimestampErrorFilter())
    logger.addFilter(TextProcessorErrorFilter())
    logger.addFilter(LPCProcessorErrorFilter())
    logger.addFilter(CollatedErrorFilter())
