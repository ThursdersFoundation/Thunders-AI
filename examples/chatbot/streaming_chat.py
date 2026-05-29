"""Streaming chat example using Thunders AI.

Demonstrates how to receive streaming responses from the Thunders AI
chat model, displaying tokens as they arrive for a real-time experience.
"""

import sys
from thunders_ai import ThundersAI, ThundersConfig


def stream_callback(token: str, index: int) -> None:
    """Handle each streamed token as it arrives.

    Args:
        token: The generated text token.
        index: The token index in the stream.
    """
    # Print each token without a newline for real-time effect
    sys.stdout.write(token)
    sys.stdout.flush()


def main() -> None:
    """Run the streaming chat example."""
    # --- Example 1: Basic streaming ---
    # Tokens are delivered one at a time via the callback
    config = ThundersConfig(
        model="thunders-7b",
        temperature=0.7,
        max_tokens=512,
    )
    ai = ThundersAI(config=config)

    print("=== Basic Streaming Chat ===")
    print("User: Tell me a short story about a robot discovering music.")
    print("AI:   ", end="")

    # Stream the response token by token
    full_response = ai.chat_stream(
        "Tell me a short story about a robot discovering music.",
        callback=stream_callback,
    )
    print("\n")
    print(f"Total tokens received: {len(full_response.split())}")
    print()

    # --- Example 2: Streaming with progress tracking ---
    # Track streaming progress with a progress bar
    config = ThundersConfig(model="thunders-7b", max_tokens=256)
    ai = ThundersAI(config=config)

    print("=== Streaming with Progress ===")
    print("User: Explain how neural networks learn.")

    token_count = 0
    def progress_callback(token: str, index: int) -> None:
        """Track and display streaming progress."""
        nonlocal token_count
        token_count += 1
        sys.stdout.write(token)
        if token_count % 20 == 0:
            sys.stdout.write(f" [{token_count} tokens]")
        sys.stdout.flush()

    print("AI:   ", end="")
    response = ai.chat_stream(
        "Explain how neural networks learn.",
        callback=progress_callback,
    )
    print(f"\nStream complete. Total tokens: {token_count}")
    print()

    # --- Example 3: Streaming multiple queries ---
    # Send multiple prompts in sequence with streaming
    prompts = [
        "What is reinforcement learning?",
        "How is it used in robotics?",
    ]

    print("=== Sequential Streaming ===")
    for prompt in prompts:
        print(f"User: {prompt}")
        print("AI:   ", end="")
        ai.chat_stream(prompt, callback=stream_callback)
        print("\n")


if __name__ == "__main__":
    main()
