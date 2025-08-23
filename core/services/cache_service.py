"""
Redis-based caching service for the RAG application.
Handles embedding caching, query result caching, and session management.
"""
from __future__ import annotations
import redis
import json
import hashlib
import logging
import time
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based caching service for RAG application."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis connection."""
        settings = get_settings()
        self.redis_url = redis_url or settings.redis.url
        self.embedding_ttl = settings.redis.embedding_cache_ttl
        self.query_ttl = settings.redis.query_cache_ttl
        self.session_ttl = settings.redis.session_ttl
        
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                max_connections=settings.redis.max_connections,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis connection established: {self.redis_url}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except redis.ConnectionError:
            return False
    
    # ========== EMBEDDING CACHING ==========
    
    def _embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"embedding:{text_hash}"
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding for text."""
        if not self.is_available():
            return None
        
        try:
            cache_key = self._embedding_cache_key(text)
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for embedding (saved OpenAI API call)")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error retrieving cached embedding: {e}")
        
        return None
    
    def cache_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding result."""
        if not self.is_available():
            return False
        
        try:
            cache_key = self._embedding_cache_key(text)
            self.redis_client.setex(
                cache_key, 
                self.embedding_ttl, 
                json.dumps(embedding)
            )
            logger.debug(f"Cached embedding for text ({len(text)} chars)")
            return True
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
            return False
    
    # ========== QUERY RESULT CACHING ==========
    
    def _query_cache_key(self, query: str, top_k: int, config_hash: str) -> str:
        """Generate cache key for query results."""
        content = f"{query}:{top_k}:{config_hash}"
        query_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"query_result:{query_hash}"
    
    def _config_hash(self, config) -> str:
        """Generate hash for hybrid search configuration."""
        config_str = f"{config.bm25_weight}:{config.vector_weight}:{config.bm25_top_k}:{config.vector_top_k}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_cached_query_result(self, query: str, top_k: int, config) -> Optional[tuple]:
        """Retrieve cached query result."""
        if not self.is_available():
            return None
        
        try:
            config_hash = self._config_hash(config)
            cache_key = self._query_cache_key(query, top_k, config_hash)
            cached = self.redis_client.get(cache_key)
            
            if cached:
                logger.info(f"Cache hit for query: {query[:50]}...")
                data = json.loads(cached)
                context_df = pd.read_json(data['context'])
                return context_df, data['response']
        except Exception as e:
            logger.warning(f"Error retrieving cached query result: {e}")
        
        return None
    
    def cache_query_result(self, query: str, top_k: int, config, 
                          context_df: pd.DataFrame, response: Any) -> bool:
        """Cache query result."""
        if not self.is_available():
            return False
        
        try:
            config_hash = self._config_hash(config)
            cache_key = self._query_cache_key(query, top_k, config_hash)
            
            # Convert response to serializable format
            if hasattr(response, 'model_dump'):
                response_data = response.model_dump()
            else:
                response_data = response
            
            cache_data = {
                'context': context_df.to_json(),
                'response': response_data,
                'cached_at': time.time()
            }
            
            self.redis_client.setex(
                cache_key, 
                self.query_ttl, 
                json.dumps(cache_data)
            )
            logger.info(f"Cached query result for: {query[:50]}...")
            return True
        except Exception as e:
            logger.warning(f"Error caching query result: {e}")
            return False
    
    # ========== SESSION MANAGEMENT ==========
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new user session."""
        import uuid
        session_id = str(uuid.uuid4())
        
        session_data = {
            'user_id': user_id or 'anonymous',
            'created_at': time.time(),
            'query_history': [],
            'total_queries': 0,
            'preferences': {}
        }
        
        if self.is_available():
            try:
                self.redis_client.setex(
                    f"session:{session_id}", 
                    self.session_ttl, 
                    json.dumps(session_data)
                )
                logger.info(f"Created session: {session_id}")
            except Exception as e:
                logger.warning(f"Error creating session: {e}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data."""
        if not self.is_available():
            return None
        
        try:
            cached = self.redis_client.get(f"session:{session_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error retrieving session: {e}")
        
        return None
    
    def update_session(self, session_id: str, query: str, response_summary: str) -> bool:
        """Add query to session history."""
        if not self.is_available():
            return False
        
        try:
            session_data = self.get_session(session_id)
            if session_data:
                # Add to history (keep last 10 queries)
                session_data['query_history'].append({
                    'query': query[:100],  # Truncate long queries
                    'summary': response_summary[:200],  # Truncate long summaries
                    'timestamp': time.time()
                })
                session_data['query_history'] = session_data['query_history'][-10:]
                session_data['total_queries'] += 1
                
                # Update in Redis
                self.redis_client.setex(
                    f"session:{session_id}", 
                    self.session_ttl, 
                    json.dumps(session_data)
                )
                return True
        except Exception as e:
            logger.warning(f"Error updating session: {e}")
        
        return False
    
    # ========== CACHE MANAGEMENT ==========
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.is_available():
            return {'error': 'Redis not available'}
        
        try:
            info = self.redis_client.info()
            
            # Count keys by type
            embedding_keys = len(self.redis_client.keys("embedding:*"))
            query_keys = len(self.redis_client.keys("query_result:*"))
            session_keys = len(self.redis_client.keys("session:*"))
            
            return {
                'redis_connected': True,
                'memory_used_mb': round(info['used_memory'] / 1024 / 1024, 2),
                'total_keys': info['db0']['keys'] if 'db0' in info else 0,
                'embedding_cache_count': embedding_keys,
                'query_cache_count': query_keys,
                'active_sessions': session_keys,
                'uptime_seconds': info['uptime_in_seconds']
            }
        except Exception as e:
            return {'error': f'Failed to get stats: {e}'}
    
    def clear_cache(self, cache_type: str = 'all') -> bool:
        """Clear specific cache type or all caches."""
        if not self.is_available():
            return False
        
        try:
            if cache_type == 'embeddings':
                keys = self.redis_client.keys("embedding:*")
            elif cache_type == 'queries':
                keys = self.redis_client.keys("query_result:*")
            elif cache_type == 'sessions':
                keys = self.redis_client.keys("session:*")
            elif cache_type == 'all':
                self.redis_client.flushdb()
                logger.info("Cleared all Redis cache")
                return True
            else:
                return False
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} {cache_type} cache entries")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


# Global cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service