"""
Service utility functions for checking FastRay API status and health.
"""
import httpx
import asyncio
from typing import Dict, List, Optional
from datetime import datetime


async def check_service_status(api_url: str = "http://oppenheimer.neurology.emory.edu:8000") -> dict:
    """
    Check FastRay service status and capacity.
    
    Args:
        api_url: Base URL for FastRay API
    
    Returns:
        dict with service configuration and stats
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get config
            config_response = await client.get(f"{api_url}/config")
            config_response.raise_for_status()
            config = config_response.json()
            
            # Get stats
            stats_response = await client.get(f"{api_url}/stats")
            stats_response.raise_for_status()
            stats = stats_response.json()
            
            return {
                "config": config,
                "stats": stats,
                "status": "healthy"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def print_service_status(status_info: dict):
    """Print service status in a readable format"""
    if status_info.get("status") == "error":
        print(f"⚠️  Could not check service status: {status_info.get('error')}")
        return
    
    config = status_info.get("config", {})
    stats = status_info.get("stats", {})
    
    print("\n" + "="*60)
    print("FastRay Service Status")
    print("="*60)
    
    # Configuration info
    num_replicas = config.get("num_replicas", "unknown")
    max_ongoing_requests = config.get("max_ongoing_requests_per_replica", "unknown")
    total_capacity = num_replicas * max_ongoing_requests if isinstance(num_replicas, int) and isinstance(max_ongoing_requests, int) else "unknown"
    
    print(f"\n📊 Configuration:")
    print(f"   Replicas: {num_replicas}")
    print(f"   Max concurrent requests per replica: {max_ongoing_requests}")
    print(f"   Total service capacity: ~{total_capacity} concurrent requests")
    
    # Stats info
    stats_summary = stats.get("summary", {})
    total_requests = stats_summary.get("total_requests", 0)
    requests_per_replica_avg = stats_summary.get("requests_per_replica_avg", 0)
    load_balance_ratio = stats_summary.get("load_balance_ratio", 0)
    
    print(f"\n📈 Request Statistics:")
    print(f"   Total requests handled: {total_requests}")
    print(f"   Avg requests per replica: {requests_per_replica_avg}")
    print(f"   Load balance ratio: {load_balance_ratio} (closer to 1.0 = better)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if isinstance(total_capacity, int):
        recommended_concurrent = min(total_capacity * 0.7, 50)  # Use 70% of capacity, max 50
        print(f"   Recommended max_concurrent_requests: {int(recommended_concurrent)}")
        if load_balance_ratio > 2.0:
            print(f"   ⚠️  Load is unevenly distributed (ratio > 2.0)")
        if requests_per_replica_avg > max_ongoing_requests * 0.8:
            print(f"   ⚠️  Replicas are near capacity")
        else:
            print(f"   ✓ Service has capacity for more concurrent requests")
    
    print("="*60 + "\n")


async def get_realtime_stats(api_url: str = "http://oppenheimer.neurology.emory.edu:8000") -> dict:
    """
    Get real-time service statistics including current load.
    
    Args:
        api_url: Base URL for FastRay API
    
    Returns:
        dict with current service load and capacity metrics
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            stats_response = await client.get(f"{api_url}/stats")
            stats_response.raise_for_status()
            stats = stats_response.json()
            
            config_response = await client.get(f"{api_url}/config")
            config_response.raise_for_status()
            config = config_response.json()
            
            # Calculate current utilization
            # The /config endpoint returns num_replicas_configured or num_replicas_running
            num_replicas = config.get("num_replicas_running") or config.get("num_replicas_configured") or config.get("num_replicas", 30)
            max_per_replica = 3  # Not in config response, but we know it's 3 from deployment (.env MAX_ONGOING_REQUESTS=3)
            total_capacity = num_replicas * max_per_replica
            
            summary = stats.get("summary", {})
            avg_requests = summary.get("requests_per_replica_avg", 0)
            max_requests = summary.get("requests_per_replica_max", 0)
            
            # Note: /stats returns cumulative request counts, not current active requests
            # So this is historical data, not real-time load
            # Estimate current load (this is approximate based on recent requests)
            current_load_estimate = max_requests  # Use max as conservative estimate
            # Utilization should be based on total capacity, not per-replica capacity
            utilization_pct = (current_load_estimate / total_capacity * 100) if total_capacity > 0 else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "stats": stats,
                "metrics": {
                    "total_capacity": total_capacity,
                    "current_load_estimate": current_load_estimate,
                    "avg_requests_per_replica": avg_requests,
                    "max_requests_per_replica": max_requests,
                    "utilization_percent": round(utilization_pct, 1),
                    "load_balance_ratio": summary.get("load_balance_ratio", 0),
                    "all_replicas_used": stats.get("interpretation", {}).get("if_all_replicas_used", False)
                }
            }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def print_realtime_stats(stats: dict):
    """Print real-time service statistics"""
    if "error" in stats:
        print(f"⚠️  Could not get real-time stats: {stats['error']}")
        return
    
    metrics = stats.get("metrics", {})
    print(f"\n📊 Service Load Statistics:")
    print(f"   Capacity: {metrics.get('total_capacity', '?')} concurrent requests")
    print(f"   Note: /stats shows cumulative request counts (historical), not current active requests")
    print(f"   Max requests per replica (historical): {metrics.get('max_requests_per_replica', '?')}")
    print(f"   Avg requests per replica (historical): {metrics.get('avg_requests_per_replica', '?')}")
    print(f"   Load balance: {metrics.get('load_balance_ratio', '?')} (1.0 = perfect)")
    
    # Utilization is now calculated correctly (based on total capacity)
    utilization = metrics.get("utilization_percent", 0)
    if utilization > 80:
        print(f"   ⚠️  High utilization ({utilization:.1f}%) - but note this is historical, not current")
    elif utilization < 50:
        print(f"   ✓ Low utilization ({utilization:.1f}%) - can likely increase concurrency")
    else:
        print(f"   Utilization: {utilization:.1f}% (historical data)")
    
    if not metrics.get("all_replicas_used", False):
        print(f"   ⚠️  Not all replicas handled requests (historical) - may indicate load imbalance")


class ConcurrencyMonitor:
    """
    Monitor transcription performance to help determine optimal concurrency.
    Tracks response times, error rates, and service load.
    """
    def __init__(self, api_url: str = "http://oppenheimer.neurology.emory.edu:8000"):
        self.api_url = api_url
        self.response_times: List[float] = []  # Client-side: includes network, queue, etc.
        self.inference_times: List[float] = []  # Server-side: actual GPU inference time
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.last_status_check = None
        
    def start(self):
        """Start monitoring"""
        self.start_time = datetime.now()
        self.response_times = []
        self.inference_times = []
        self.error_count = 0
        self.success_count = 0
        
    def record_response(self, response_time: float, success: bool = True, inference_time: float = None):
        """Record a transcription response
        
        Args:
            response_time: Total client-side time (includes network, queue, retries)
            success: Whether the request succeeded
            inference_time: Server-side inference time from the API response (optional)
        """
        if success:
            self.success_count += 1
            self.response_times.append(response_time)
            if inference_time is not None:
                self.inference_times.append(inference_time)
        else:
            self.error_count += 1
    
    async def check_service_status(self) -> dict:
        """Check current service status"""
        self.last_status_check = await get_realtime_stats(self.api_url)
        return self.last_status_check
    
    def get_summary(self) -> dict:
        """Get monitoring summary"""
        import statistics
        
        if not self.response_times:
            return {
                "total_requests": self.success_count + self.error_count,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "error_rate": (self.error_count / (self.success_count + self.error_count) * 100) if (self.success_count + self.error_count) > 0 else 0,
                "avg_response_time": None,
                "avg_inference_time": None,
            }
        
        # Filter outliers for response times (remove top 1% and bottom 1% to handle startup/weird cases)
        sorted_response_times = sorted(self.response_times)
        if len(sorted_response_times) > 20:
            # Remove outliers: top 1% and bottom 1%
            trim_count = max(1, int(len(sorted_response_times) * 0.01))
            filtered_response_times = sorted_response_times[trim_count:-trim_count] if trim_count > 0 else sorted_response_times
        else:
            filtered_response_times = sorted_response_times
        
        result = {
            "total_requests": self.success_count + self.error_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": round((self.error_count / (self.success_count + self.error_count)) * 100, 2) if (self.success_count + self.error_count) > 0 else 0,
            "avg_response_time": round(statistics.mean(filtered_response_times), 3),
            "median_response_time": round(statistics.median(filtered_response_times), 3),
            "min_response_time": round(min(filtered_response_times), 3),
            "max_response_time": round(max(filtered_response_times), 3),
            "p95_response_time": round(sorted_response_times[int(len(sorted_response_times) * 0.95)] if len(sorted_response_times) > 0 else 0, 3),
        }
        
        # Add inference time stats if available
        if self.inference_times:
            sorted_inference_times = sorted(self.inference_times)
            if len(sorted_inference_times) > 20:
                trim_count = max(1, int(len(sorted_inference_times) * 0.01))
                filtered_inference_times = sorted_inference_times[trim_count:-trim_count] if trim_count > 0 else sorted_inference_times
            else:
                filtered_inference_times = sorted_inference_times
            
            result.update({
                "avg_inference_time": round(statistics.mean(filtered_inference_times), 3),
                "median_inference_time": round(statistics.median(filtered_inference_times), 3),
                "min_inference_time": round(min(filtered_inference_times), 3),
                "max_inference_time": round(max(filtered_inference_times), 3),
                "p95_inference_time": round(sorted_inference_times[int(len(sorted_inference_times) * 0.95)] if len(sorted_inference_times) > 0 else 0, 3),
                "network_overhead": round(statistics.mean(filtered_response_times) - statistics.mean(filtered_inference_times), 3) if len(filtered_inference_times) > 0 else None,
            })
        else:
            result["avg_inference_time"] = None
        
        return result
    
    def print_summary(self):
        """Print monitoring summary"""
        summary = self.get_summary()
        print(f"\n📈 Performance Summary:")
        print(f"   Total requests: {summary['total_requests']}")
        print(f"   Success: {summary['success_count']} ({100 - summary['error_rate']:.1f}%)")
        print(f"   Errors: {summary['error_count']} ({summary['error_rate']:.1f}%)")
        
        # Focus on inference time (actual processing) - this is what matters
        if summary.get('avg_inference_time'):
            print(f"\n   ⚡ Server-side inference times (actual GPU processing - this is what matters):")
            print(f"     Avg: {summary['avg_inference_time']}s")
            print(f"     Median: {summary['median_inference_time']}s")
            print(f"     P95: {summary['p95_inference_time']}s")
            print(f"     Range: {summary['min_inference_time']}s - {summary['max_inference_time']}s")
        
        # Show client-side times but de-emphasize (queue wait doesn't count)
        if summary['avg_response_time']:
            print(f"\n   📊 Client-side response times (includes queue wait - less relevant with high concurrency):")
            print(f"     Avg: {summary['avg_response_time']}s")
            if summary.get('avg_inference_time') and summary.get('network_overhead') is not None:
                overhead = summary['network_overhead']
                if overhead > 0:
                    print(f"     Queue/network overhead: ~{overhead:.3f}s (waiting in queue)")
                elif overhead < 0:
                    print(f"     ⚠️  Clock sync issue? Inference time > response time")
        
        # Recommendations based on inference time (not total response time)
        if summary['error_rate'] > 5:
            print(f"\n   ⚠️  High error rate ({summary['error_rate']:.1f}%) - reduce concurrency")
        elif summary['error_rate'] < 1 and summary.get('avg_inference_time'):
            avg_inf = summary['avg_inference_time']
            if avg_inf < 2.0:
                print(f"\n   ✓ Low error rate and fast inference ({avg_inf:.2f}s avg) - can likely increase concurrency")
            elif avg_inf > 5.0:
                print(f"\n   ⚠️  Slow inference ({avg_inf:.2f}s avg) - may be overloaded, consider reducing concurrency")
        
        if self.last_status_check and "metrics" in self.last_status_check:
            metrics = self.last_status_check["metrics"]
            if metrics.get("utilization_percent", 0) < 60:
                print(f"   ✓ Service utilization at {metrics['utilization_percent']:.1f}% - room for more concurrency")

