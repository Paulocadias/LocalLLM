#!/bin/bash
#
# Automatic Background Process Cleanup Script
# Cleans up all background bash processes stored in Redis
#

echo "🧹 Background Process Cleanup Tool"
echo "=================================="

# Check if Redis is available
if ! docker exec localllm-redis redis-cli PING > /dev/null 2>&1; then
    echo "❌ Error: Redis is not available"
    exit 1
fi

echo "✅ Redis connection OK"
echo ""

# Get all bash process keys from Redis
echo "🔍 Scanning for background processes..."
KEYS=$(docker exec localllm-redis redis-cli KEYS "bash:*" 2>/dev/null)

if [ -z "$KEYS" ]; then
    echo "✨ No background processes found - all clean!"
    exit 0
fi

# Count the keys
COUNT=$(echo "$KEYS" | wc -l)
echo "📊 Found $COUNT background processes"
echo ""

# Delete all bash process keys
echo "🗑️  Deleting background processes..."
docker exec localllm-redis redis-cli --raw KEYS "bash:*" | xargs -I {} docker exec localllm-redis redis-cli DEL {}

# Verify cleanup
REMAINING=$(docker exec localllm-redis redis-cli KEYS "bash:*" 2>/dev/null | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ Successfully cleaned up $COUNT background processes"
else
    echo "⚠️  Warning: $REMAINING processes remain"
fi

echo ""
echo "✨ Cleanup complete!"
