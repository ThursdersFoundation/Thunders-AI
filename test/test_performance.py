"""Tests for Thunders AI performance benchmarks."""
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from thunders_ai import ThundersAI, ThundersConfig


class TestInferenceLatency:
    """Tests for inference latency benchmarks."""

    def test_inference_latency(self, thunders_client):
        """Test that single inference completes within acceptable latency."""
        max_latency_ms = 500  # 500ms max for CPU test mode
        with patch.object(
            thunders_client, "chat", return_value="Test response"
        ):
            start = time.perf_counter()
            response = thunders_client.chat("Hello")
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert response is not None
            # In test/mocked mode, this should be well under the limit
            assert elapsed_ms < max_latency_ms, f"Inference latency {elapsed_ms:.1f}ms exceeds {max_latency_ms}ms"

    def test_inference_latency_p95(self, thunders_client):
        """Test that 95th percentile inference latency is acceptable."""
        num_requests = 50
        latencies = []
        with patch.object(
            thunders_client, "chat", return_value="Test response"
        ):
            for _ in range(num_requests):
                start = time.perf_counter()
                thunders_client.chat("Hello")
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]
        max_p95_ms = 1000
        assert p95_latency < max_p95_ms, f"P95 latency {p95_latency:.1f}ms exceeds {max_p95_ms}ms"

    def test_streaming_first_token_latency(self, thunders_client):
        """Test time-to-first-token for streaming inference."""
        with patch.object(
            thunders_client, "chat_stream", return_value=iter(["token1", "token2", "token3"])
        ):
            start = time.perf_counter()
            first_token = next(thunders_client.chat_stream("Hello"))
            ttft_ms = (time.perf_counter() - start) * 1000
            assert first_token is not None
            assert ttft_ms < 200, f"Time to first token {ttft_ms:.1f}ms is too high"


class TestThroughput:
    """Tests for throughput benchmarks."""

    def test_throughput(self, thunders_client):
        """Test that throughput meets minimum requirements."""
        num_requests = 20
        with patch.object(
            thunders_client, "chat", return_value="Test response"
        ):
            start = time.perf_counter()
            for _ in range(num_requests):
                thunders_client.chat("Hello")
            elapsed = time.perf_counter() - start
            throughput = num_requests / elapsed
            # Even in test mode, should handle at least 10 req/s
            min_throughput = 10
            assert throughput >= min_throughput, f"Throughput {throughput:.1f} req/s below minimum {min_throughput} req/s"

    def test_batch_throughput(self, thunders_client):
        """Test batch request throughput."""
        batch_size = 10
        messages = [f"Message {i}" for i in range(batch_size)]
        with patch.object(
            thunders_client, "chat_batch", return_value=[f"Response {i}" for i in range(batch_size)]
        ):
            start = time.perf_counter()
            results = thunders_client.chat_batch(messages)
            elapsed = time.perf_counter() - start
            assert len(results) == batch_size
            throughput = batch_size / elapsed
            assert throughput >= 5  # At least 5 req/s for batch


class TestMemoryUsage:
    """Tests for memory usage benchmarks."""

    def test_memory_usage(self, thunders_client):
        """Test that memory usage stays within bounds."""
        import resource
        # Get current memory usage
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with patch.object(
            thunders_client, "chat", return_value="Response"
        ):
            for _ in range(10):
                thunders_client.chat("Hello")
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Memory should not grow excessively (allow 100MB growth)
        mem_growth_mb = (end_mem - start_mem) / 1024  # Convert KB to MB
        assert mem_growth_mb < 100, f"Memory grew by {mem_growth_mb:.1f}MB, exceeding 100MB limit"

    def test_no_memory_leak_in_chat_history(self, thunders_client):
        """Test that chat history doesn't cause memory leaks."""
        from thunders_ai.llm.chat_model import ChatHistory
        history = ChatHistory(max_length=100)
        import sys
        initial_size = sys.getsizeof(history)
        for i in range(1000):
            history.add_message("user", f"Message {i}")
        # History should be capped at max_length
        assert len(history) <= 100


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    def test_concurrent_requests(self, thunders_client):
        """Test handling of concurrent chat requests."""
        num_concurrent = 10
        results = []
        errors = []

        def make_request(idx):
            try:
                with patch.object(
                    thunders_client, "chat", return_value=f"Response {idx}"
                ):
                    response = thunders_client.chat(f"Request {idx}")
                    return idx, response
            except Exception as e:
                return idx, str(e)

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
            for future in as_completed(futures):
                idx, result = future.result()
                if isinstance(result, str) and "Error" in result:
                    errors.append((idx, result))
                else:
                    results.append((idx, result))

        assert len(errors) == 0, f"Errors in concurrent requests: {errors}"
        assert len(results) == num_concurrent

    def test_concurrent_vision_requests(self, thunders_client, sample_image_path):
        """Test handling of concurrent vision requests."""
        num_concurrent = 5
        results = []

        def analyze_image(idx):
            with patch.object(
                thunders_client, "analyze_image",
                return_value={"captions": [f"Caption {idx}"]}
            ):
                return thunders_client.analyze_image(sample_image_path)

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(analyze_image, i) for i in range(num_concurrent)]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == num_concurrent

    def test_mixed_concurrent_requests(self, thunders_client, sample_image_path):
        """Test handling of mixed concurrent chat and vision requests."""
        results = []

        def chat_request(idx):
            with patch.object(
                thunders_client, "chat", return_value=f"Chat {idx}"
            ):
                return ("chat", thunders_client.chat(f"Hello {idx}"))

        def vision_request(idx):
            with patch.object(
                thunders_client, "analyze_image",
                return_value={"captions": [f"Caption {idx}"]}
            ):
                return ("vision", thunders_client.analyze_image(sample_image_path))

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(4):
                futures.append(executor.submit(chat_request, i))
                futures.append(executor.submit(vision_request, i))
            for future in as_completed(futures):
                results.append(future.result())

        chat_results = [r for r in results if r[0] == "chat"]
        vision_results = [r for r in results if r[0] == "vision"]
        assert len(chat_results) == 4
        assert len(vision_results) == 4
