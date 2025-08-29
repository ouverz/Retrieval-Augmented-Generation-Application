# Dependency Injection Migration Guide

This document outlines the migration from global state anti-pattern to thread-safe dependency injection in the RAG application.

## Problem Statement

The original dependency injection system in `backend/dependencies.py` used a global state anti-pattern:

```python
# PROBLEMATIC CODE (now deprecated)
_global_container = None

def app_container(dsn: str = Depends(pg_dsn)) -> AppContainer:
    global _global_container
    if _global_container is None:
        _global_container = AppContainer(data_dir=data_dir, pg_dsn=dsn)
    return _global_container
```

### Issues:
- **Thread Safety Violation**: Race conditions in multi-threaded environments
- **Testing Difficulties**: Impossible to isolate tests with global state
- **Mutable State**: Container could be modified during request processing
- **Hidden Dependencies**: Environment variables accessed implicitly

## Solution: Immutable Container Architecture

### New Architecture Components

#### 1. Immutable Container (`backend/container.py`)
```python
@dataclass(frozen=True)
class ImmutableAppContainer:
    """Immutable container with all services initialized upfront."""
    data_dir: str
    pg_dsn: str
    cache_service: CacheService
    llm_factory: LLMFactory
    bm25_engine: BM25SearchEngine
    vector_engine: VectorSearchEngine
    hybrid_engine: HybridSearchEngine
```

#### 2. Thread-Safe Factory (`backend/dependencies.py`)
```python
def get_immutable_container(dsn: str = Depends(pg_dsn)) -> ImmutableAppContainer:
    """Create new immutable container instance - no global state."""
    data_dir = os.getenv("DATA_DIR", "./data/documents")
    return create_immutable_container(data_dir=data_dir, pg_dsn=dsn)

# Type annotation for FastAPI dependency injection
ImmutableContainerDep = Annotated[ImmutableAppContainer, Depends(get_immutable_container)]
```

#### 3. Updated Endpoints
New `/v2` endpoints demonstrate the thread-safe pattern:
- `/health/v2` - Thread-safe health check
- `/cache/v2/stats` - Thread-safe cache statistics  
- `/cache/v2/health` - Thread-safe cache health
- `/query/v2` - Thread-safe query processing

## Migration Status

### ✅ Completed
- **Phase 2A**: Immutable container architecture implemented
- **Phase 2B**: Thread-safe endpoints with side-by-side comparison
- **Phase 2C**: Comprehensive test suite and documentation

### 🔄 In Progress  
- Gradual migration of remaining endpoints to use `ImmutableContainerDep`
- Deprecation warnings added to global state functions

### 📋 Remaining Work
- Migrate `/init` router to new architecture
- Migrate `/ingest` router to new architecture  
- Remove deprecated global state functions
- Update all existing endpoints to use v2 pattern

## Usage Examples

### Old Pattern (Deprecated)
```python
@router.get("/stats")
def get_stats(container: AppContainer = Depends(get_app_container)):
    # Uses global state - NOT thread-safe
    return container.cache_service.get_cache_stats()
```

### New Pattern (Recommended)
```python
@router.get("/v2/stats")
def get_stats_v2(container: ImmutableContainerDep):
    # Thread-safe, immutable, testable
    return {
        "stats": container.cache_service.get_cache_stats(),
        "container_type": "immutable", 
        "thread_safe": True
    }
```

## Testing Benefits

### Before (Global State Issues)
- Tests affected each other due to shared global state
- Difficult to mock dependencies
- Required external services for testing
- Cannot run tests in parallel

### After (Thread-Safe DI)
```python
def test_endpoint_with_mock():
    mock_container = Mock(spec=ImmutableAppContainer)
    mock_container.is_ready.return_value = True
    
    app.dependency_overrides[get_immutable_container] = lambda: mock_container
    client = TestClient(app)
    
    response = client.get("/cache/v2/stats")
    assert response.status_code == 200
```

## Performance Impact

### Thread Safety Test Results
- ✅ 10 concurrent threads creating containers simultaneously
- ✅ All containers are unique instances (no shared state)
- ✅ All containers properly immutable (frozen dataclass)
- ✅ No race conditions or state corruption detected

### Memory Usage
- Immutable containers prevent accidental state modification
- Services initialized once per container instance
- No memory leaks from global state accumulation

## Benefits Achieved

### 🔒 Thread Safety
- No shared mutable state between requests
- Safe concurrent request handling
- Race condition elimination

### 🧪 Testability  
- Easy dependency mocking with `app.dependency_overrides`
- Isolated test environments
- Parallel test execution capability

### 🛡️ Immutability
- Container state cannot be accidentally modified
- Prevents service corruption during request processing
- Predictable behavior across request lifecycle

### 📊 Observability
- Clear labeling of container type in responses
- Thread safety status indication
- Better error handling and logging

## Migration Timeline

- **Phase 2A** (Completed): Foundation architecture
- **Phase 2B** (Completed): Router demonstrations with v2 endpoints
- **Phase 2C** (Completed): Testing infrastructure and documentation
- **Phase 2D** (Next): Complete router migration
- **Phase 2E** (Final): Remove deprecated global state system

This migration represents a fundamental improvement in application architecture, moving from problematic global state to production-ready thread-safe dependency injection.