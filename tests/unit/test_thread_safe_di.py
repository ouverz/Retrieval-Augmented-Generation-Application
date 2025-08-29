#!/usr/bin/env python3
"""
Unit tests demonstrating the new thread-safe dependency injection system.
Shows how the immutable container architecture enables proper testing.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from backend.dependencies import get_immutable_container, ImmutableContainerDep
from backend.container import ImmutableAppContainer, create_immutable_container
from backend.routers.cache import router as cache_router
from backend.routers.query import router as query_router
from backend.main import app


class TestThreadSafeDependencyInjection:
    """Test suite for thread-safe dependency injection system."""
    
    def test_container_isolation(self):
        """Test that multiple container instances are properly isolated."""
        container1 = get_immutable_container("postgresql://test1:test@localhost/test1")
        container2 = get_immutable_container("postgresql://test2:test@localhost/test2")
        
        # Containers should be different instances
        assert container1 is not container2
        assert container1.pg_dsn != container2.pg_dsn
        assert container1.data_dir == container2.data_dir  # Same default data_dir
        
    def test_container_immutability(self):
        """Test that containers are properly immutable."""
        container = get_immutable_container("postgresql://test:test@localhost/test")
        
        # Should not be able to modify fields
        with pytest.raises(Exception):  # dataclass frozen exception
            container.pg_dsn = "modified"
            
        with pytest.raises(Exception):
            container.data_dir = "modified"
            
        # Verify frozen attribute
        assert container.__dataclass_params__.frozen is True
        
    def test_container_services_initialized(self):
        """Test that all required services are properly initialized."""
        container = get_immutable_container("postgresql://test:test@localhost/test")
        
        assert container.cache_service is not None
        assert container.llm_factory is not None
        assert container.bm25_engine is not None
        assert container.vector_engine is not None
        assert container.hybrid_engine is not None
        
    def test_dependency_injection_mocking(self):
        """Test that dependencies can be easily mocked for testing."""
        # Create a mock container
        mock_container = Mock(spec=ImmutableAppContainer)
        mock_container.is_ready.return_value = True
        mock_container.get_cache_health.return_value = {"status": "ok"}
        
        # Create test app with dependency override
        test_app = FastAPI()
        test_app.include_router(cache_router, prefix="/cache")
        
        # Override the dependency
        test_app.dependency_overrides[get_immutable_container] = lambda: mock_container
        
        # Test the endpoint
        client = TestClient(test_app)
        response = client.get("/cache/v2/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["container_type"] == "immutable"
        assert data["thread_safe"] is True
        
        # Verify mock was called
        mock_container.get_cache_health.assert_called_once()
        
        # Cleanup
        test_app.dependency_overrides.clear()
        
    def test_cache_endpoint_v2_with_mock(self):
        """Test v2 cache endpoint with mocked dependencies."""
        mock_container = Mock(spec=ImmutableAppContainer)
        mock_container.is_ready.return_value = True
        mock_container.cache_service.get_cache_stats.return_value = {
            "redis_connected": True,
            "memory_used_mb": 10.5,
            "total_keys": 100
        }
        
        test_app = FastAPI()
        test_app.include_router(cache_router, prefix="/cache")
        test_app.dependency_overrides[get_immutable_container] = lambda: mock_container
        
        client = TestClient(test_app)
        response = client.get("/cache/v2/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["container_type"] == "immutable"
        assert data["thread_safe"] is True
        assert data["container_ready"] is True
        assert "stats" in data
        
        # Cleanup
        test_app.dependency_overrides.clear()
        
    def test_health_endpoint_v2_with_mock(self):
        """Test v2 health endpoint with mocked dependencies."""
        mock_container = Mock(spec=ImmutableAppContainer)
        mock_container.is_ready.return_value = True
        mock_container.get_cache_health.return_value = {
            "redis_connected": True,
            "cache_available": True
        }
        
        # Test using the main app with dependency override
        app.dependency_overrides[get_immutable_container] = lambda: mock_container
        
        client = TestClient(app)
        response = client.get("/health/v2")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["container_type"] == "immutable"
        assert data["thread_safe"] is True
        
        # Cleanup
        app.dependency_overrides.clear()
        
    def test_concurrent_container_creation(self):
        """Test that concurrent container creation is thread-safe."""
        import threading
        import concurrent.futures
        
        containers = []
        container_ids = []
        
        def create_container(thread_id):
            container = get_immutable_container(f"postgresql://thread{thread_id}:test@localhost/test")
            return container, id(container)
        
        # Create containers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_container, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
        containers, container_ids = zip(*results)
        
        # All containers should be different instances
        assert len(set(container_ids)) == len(container_ids)
        
        # All containers should be immutable
        assert all(c.__dataclass_params__.frozen for c in containers)
        
        # All containers should have different DSNs
        dsns = [c.pg_dsn for c in containers]
        assert len(set(dsns)) == len(dsns)


class TestBackwardCompatibility:
    """Test that old global state system still works during migration."""
    
    def test_old_endpoints_still_work(self):
        """Test that existing endpoints using global state still function."""
        client = TestClient(app)
        
        # Test old health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test old cache stats endpoint  
        response = client.get("/cache/stats")
        assert response.status_code in [200, 503]  # 503 if cache not available
        
    def test_both_systems_coexist(self):
        """Test that both old and new dependency injection systems work."""
        client = TestClient(app)
        
        # Test old health endpoint
        old_response = client.get("/health")
        assert old_response.status_code == 200
        
        # Test new health endpoint
        new_response = client.get("/health/v2")
        assert new_response.status_code == 200
        
        # Both should return valid health data
        old_data = old_response.json()
        new_data = new_response.json()
        
        assert old_data["status"] == "ok"
        assert new_data["status"] == "ok"
        assert new_data["thread_safe"] is True


if __name__ == "__main__":
    # Run tests manually for debugging
    test_instance = TestThreadSafeDependencyInjection()
    test_instance.test_container_isolation()
    test_instance.test_container_immutability() 
    test_instance.test_container_services_initialized()
    print("✅ All thread-safe dependency injection tests passed!")