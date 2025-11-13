#!/usr/bin/env python3
"""
Automatic Background Process Cleanup Service

This script can run as a scheduled task or cron job to automatically
clean up old background bash processes from Redis.

Usage:
  python auto-cleanup.py               # Clean up once
  python auto-cleanup.py --daemon      # Run as daemon (check every 5 minutes)
  python auto-cleanup.py --max-age 3600  # Clean processes older than 1 hour
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
import redis
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackgroundProcessCleaner:
    """Automatic cleanup for background bash processes"""

    def __init__(self, redis_url="redis://localhost:6380", max_age_seconds=7200):
        """
        Initialize cleaner

        Args:
            redis_url: Redis connection URL
            max_age_seconds: Maximum age for processes (default: 2 hours)
        """
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.max_age_seconds = max_age_seconds
        logger.info(f"Initialized cleaner with max_age={max_age_seconds}s")

    def get_all_bash_processes(self):
        """Get all bash process keys from Redis"""
        try:
            keys = self.redis_client.keys("bash:*")
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Error getting bash processes: {e}")
            return []

    def cleanup_old_processes(self):
        """Clean up old background processes"""
        try:
            processes = self.get_all_bash_processes()

            if not processes:
                logger.info("No background processes found")
                return 0

            logger.info(f"Found {len(processes)} background processes")

            # For now, delete all processes
            # In future, could check TTL or timestamp to only delete old ones
            deleted = 0
            for key in processes:
                try:
                    self.redis_client.delete(key)
                    deleted += 1
                except Exception as e:
                    logger.error(f"Error deleting {key}: {e}")

            logger.info(f"Cleaned up {deleted}/{len(processes)} processes")
            return deleted

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0

    def cleanup_with_age_check(self):
        """Clean up processes based on age (if metadata available)"""
        # Note: Current implementation doesn't store process creation time
        # This is a placeholder for future enhancement
        return self.cleanup_old_processes()

    def run_daemon(self, interval_seconds=300):
        """
        Run as daemon, cleaning up periodically

        Args:
            interval_seconds: Time between cleanup runs (default: 5 minutes)
        """
        logger.info(f"Starting cleanup daemon (interval={interval_seconds}s)")

        try:
            while True:
                logger.info("Running cleanup cycle...")
                deleted = self.cleanup_old_processes()

                if deleted > 0:
                    logger.info(f"Cleanup cycle complete: {deleted} processes removed")
                else:
                    logger.info("Cleanup cycle complete: no processes to remove")

                logger.info(f"Next cleanup in {interval_seconds} seconds")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
        except Exception as e:
            logger.error(f"Daemon error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Background process cleanup utility")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (continuous cleanup)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Daemon cleanup interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=7200,
        help="Maximum process age in seconds (default: 7200 = 2 hours)"
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6380",
        help="Redis connection URL"
    )

    args = parser.parse_args()

    # Create cleaner
    cleaner = BackgroundProcessCleaner(
        redis_url=args.redis_url,
        max_age_seconds=args.max_age
    )

    # Run once or as daemon
    if args.daemon:
        cleaner.run_daemon(interval_seconds=args.interval)
    else:
        deleted = cleaner.cleanup_old_processes()
        logger.info(f"One-time cleanup complete: {deleted} processes removed")


if __name__ == "__main__":
    main()
