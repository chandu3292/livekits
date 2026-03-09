#!/bin/bash

# Define log directory
LOG_DIR="./logs"
mkdir -p $LOG_DIR

echo "?? Cleaning up existing processes..."
pkill -f livekit-server
pkill -f livekit-sip
pkill -f "python3.*mcp-agent.py"
pkill -f "python3.*server.py"
pkill -f redis-server
sleep 1

echo "?? Starting all LiveKit and AI services..."

# 1. Start Redis
echo "?? Starting Redis..."
redis-server > "$LOG_DIR/redis.log" 2>&1 &
sleep 2

# 2. Start LiveKit Server
echo "?? Starting LiveKit Core..."
livekit-server --config livekit.yaml > "$LOG_DIR/livekit-server.log" 2>&1 &
sleep 2

# 3. Start LiveKit SIP Gateway
echo "?? Starting SIP Gateway..."
livekit-sip --config sip-config.yaml > "$LOG_DIR/livekit-sip.log" 2>&1 &
sleep 2

# 4. Start RAG / MCP Server
echo "?? Starting RAG Server..."
./venv/bin/python3 server.py > "$LOG_DIR/server.log" 2>&1 &
sleep 5

# 5. Start AI Agent
echo "?? Starting AI Agent..."
./venv/bin/python3 mcp-agent.py dev > "$LOG_DIR/agent.log" 2>&1 &

echo "? All systems are running in the background."
echo "?? Logs are being stored in $LOG_DIR/"
echo "?? Use 'tail -f logs/agent.log' to watch the AI's thoughts."
echo "?? To stop all services, run: pkill -f livekit && pkill -f python3"