"""Data proxy layer — intercepts MCP responses and replaces raw data with Data Profiles."""

from datavault.proxy.interceptor import DataProxyInterceptor
from datavault.proxy.profiler import DataProfiler

__all__ = ["DataProxyInterceptor", "DataProfiler"]
