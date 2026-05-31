"""WebSocket server example using Thunders AI.

Demonstrates how to connect to the Thunders AI WebSocket server for
real-time chat communication and streaming token responses.
"""

import asyncio
import json
import websockets
from websockets.asyncio.client import connect


WEBSOCKET_URL = "ws://localhost:8000/api/v1/ws/chat"
STREAM_URL = "ws://localhost:8000/api/v1/ws/stream"


async def chat_example() -> None:
    """Demonstrate real-time chat over WebSocket."""
    print("=== Real-time Chat WebSocket ===")
    print(f"Connecting to {WEBSOCKET_URL}...")

    async with connect(WEBSOCKET_URL) as ws:
        # Receive the connection confirmation message
        greeting = await ws.recv()
        data = json.loads(greeting)
        print(f"Connected! Client ID: {data.get('client_id', 'unknown')[:8]}...")
        print()

        # Send a chat message
        message = {"type": "message", "content": "Hello, Thunders AI!"}
        await ws.send(json.dumps(message))
        print(f"Sent: {message['content']}")

        # Receive the AI response
        response = await ws.recv()
        resp_data = json.loads(response)
        if resp_data.get("type") == "response":
            print(f"AI:   {resp_data.get('content', '')}")
        print()

        # Send another message
        message = {"type": "message", "content": "Tell me about robotics."}
        await ws.send(json.dumps(message))
        print(f"Sent: {message['content']}")

        response = await ws.recv()
        resp_data = json.loads(response)
        if resp_data.get("type") == "response":
            print(f"AI:   {resp_data.get('content', '')}")
        print()

        # Respond to heartbeat ping
        ping = await ws.recv()
        ping_data = json.loads(ping)
        if ping_data.get("type") == "ping":
            pong = {"type": "pong"}
            await ws.send(json.dumps(pong))
            print("Heartbeat: ping received, pong sent.")


async def stream_example() -> None:
    """Demonstrate streaming token responses over WebSocket."""
    print("\n=== Streaming Response WebSocket ===")
    print(f"Connecting to {STREAM_URL}...")

    async with connect(STREAM_URL) as ws:
        # Receive connection confirmation
        greeting = await ws.recv()
        data = json.loads(greeting)
        print(f"Connected! Client ID: {data.get('client_id', 'unknown')[:8]}...")
        print()

        # Send a streaming prompt
        prompt = {"type": "prompt", "content": "Explain neural networks", "model": "thunders-7b"}
        await ws.send(json.dumps(prompt))
        print(f"Prompt: {prompt['content']}")
        print("Streaming: ", end="")

        # Receive tokens one by one
        token_count = 0
        while True:
            response = await ws.recv()
            resp_data = json.loads(response)

            if resp_data.get("type") == "token":
                # Print each token as it arrives
                print(resp_data.get("content", ""), end=" ", flush=True)
                token_count += 1
            elif resp_data.get("type") == "done":
                # Stream completed
                print(f"\nStream complete. Total tokens: {resp_data.get('total_tokens', token_count)}")
                break
            elif resp_data.get("type") == "ping":
                # Respond to heartbeat
                await ws.send(json.dumps({"type": "pong"}))


async def multi_message_example() -> None:
    """Demonstrate sending multiple messages in sequence."""
    print("\n=== Multi-message WebSocket Chat ===")
    async with connect(WEBSOCKET_URL) as ws:
        greeting = await ws.recv()

        questions = [
            "What is machine learning?",
            "How does deep learning differ?",
            "Give me a practical example.",
        ]

        for question in questions:
            await ws.send(json.dumps({"type": "message", "content": question}))
            print(f"Q: {question}")

            response = await ws.recv()
            resp_data = json.loads(response)
            if resp_data.get("type") == "response":
                content = resp_data.get("content", "")
                print(f"A: {content[:100]}...")
            print()


async def main() -> None:
    """Run all WebSocket examples."""
    await chat_example()
    await stream_example()
    await multi_message_example()


if __name__ == "__main__":
    asyncio.run(main())
