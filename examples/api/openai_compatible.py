"""OpenAI-compatible API usage example using Thunders AI.

Demonstrates how to use Thunders AI's OpenAI-compatible API endpoints
for chat completions and streaming responses. Works as a drop-in
replacement for the OpenAI Python SDK.
"""

import openai


def main() -> None:
    """Run the OpenAI-compatible API example."""
    # --- Step 1: Configure the OpenAI client ---
    # Point the OpenAI SDK to the Thunders AI API server
    client = openai.OpenAI(
        base_url="http://localhost:8000/api/v1",  # Thunders AI server
        api_key="thunders_your_api_key_here",      # Your Thunders API key
    )
    print("OpenAI-compatible client configured for Thunders AI.")
    print(f"  Base URL: {client.base_url}")
    print()

    # --- Step 2: List available models ---
    # Retrieve the list of models available on the server
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"  - {model.id}")
    print()

    # --- Step 3: Basic chat completion ---
    # Send a simple chat message and get a response
    print("=== Basic Chat Completion ===")
    response = client.chat.completions.create(
        model="thunders-7b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.7,
        max_tokens=256,
    )
    print(f"User: What is the capital of France?")
    print(f"AI:   {response.choices[0].message.content}")
    print(f"Usage: {response.usage.total_tokens} tokens")
    print()

    # --- Step 4: Multi-turn conversation ---
    # Continue a conversation with context
    print("=== Multi-turn Conversation ===")
    messages = [
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "How do I sort a list in Python?"},
    ]
    response = client.chat.completions.create(
        model="thunders-7b",
        messages=messages,
    )
    assistant_reply = response.choices[0].message.content
    print(f"User: How do I sort a list in Python?")
    print(f"AI:   {assistant_reply}")

    # Add the assistant's reply and ask a follow-up
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "How about sorting in reverse order?"})

    response = client.chat.completions.create(
        model="thunders-7b",
        messages=messages,
    )
    print(f"User: How about sorting in reverse order?")
    print(f"AI:   {response.choices[0].message.content}")
    print()

    # --- Step 5: Streaming chat completion ---
    # Receive tokens as they are generated in real-time
    print("=== Streaming Chat Completion ===")
    print("User: Write a haiku about artificial intelligence.")
    print("AI:   ", end="")

    stream = client.chat.completions.create(
        model="thunders-7b",
        messages=[
            {"role": "user", "content": "Write a haiku about artificial intelligence."},
        ],
        stream=True,  # Enable streaming
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

    # --- Step 6: Vision analysis (OpenAI-compatible) ---
    # Analyze an image using the vision endpoint
    print("=== Vision Analysis ===")
    response = client.chat.completions.create(
        model="thunders-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/photo.jpg"},
                    },
                ],
            },
        ],
    )
    print(f"AI:   {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
