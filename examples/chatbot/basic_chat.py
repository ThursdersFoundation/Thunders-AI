"""Basic chat example using Thunders AI.

Demonstrates the simplest way to interact with the Thunders AI chat model,
including initializing the client, sending a message, and printing the
response. Also shows how to use a custom configuration.
"""

from thunders_ai import ThundersAI, ThundersConfig


def main() -> None:
    """Run the basic chat example."""
    # --- Example 1: Quick start with defaults ---
    # Uses default model and settings
    ai = ThundersAI()
    response = ai.chat("Hello, who are you?")
    print("=== Default Chat ===")
    print(f"User: Hello, who are you?")
    print(f"AI:   {response}")
    print()

    # --- Example 2: Chat with custom configuration ---
    # Configure model, temperature, and max tokens
    config = ThundersConfig(
        model="thunders-7b",
        temperature=0.8,
        max_tokens=512,
    )
    ai = ThundersAI(config=config)
    response = ai.chat("Explain quantum computing in one paragraph.")
    print("=== Custom Config Chat ===")
    print(f"User: Explain quantum computing in one paragraph.")
    print(f"AI:   {response}")
    print()

    # --- Example 3: Multi-turn conversation ---
    # ThundersAI maintains conversation context across calls
    ai = ThundersAI()
    print("=== Multi-turn Conversation ===")
    questions = [
        "What is machine learning?",
        "How does it differ from deep learning?",
        "Give me a real-world example.",
    ]
    for question in questions:
        response = ai.chat(question)
        print(f"User: {question}")
        print(f"AI:   {response}")
        print()

    # --- Example 4: System prompt ---
    # Set a system prompt to guide AI behavior
    config = ThundersConfig(
        model="thunders-7b",
        system_prompt="You are a helpful Python programming tutor. "
                      "Always provide code examples.",
    )
    ai = ThundersAI(config=config)
    response = ai.chat("How do I read a file in Python?")
    print("=== System Prompt Chat ===")
    print(f"User: How do I read a file in Python?")
    print(f"AI:   {response}")


if __name__ == "__main__":
    main()
