"""Memory-enabled chat example using Thunders AI.

Demonstrates how to use Thunders AI with conversation memory,
allowing the AI to maintain context across multiple exchanges.
Shows short-term and long-term memory management.
"""

from thunders_ai import ThundersAI, ThundersConfig
from thunders_ai.memory import ConversationMemory, LongTermMemory


def main() -> None:
    """Run the memory-enabled chat example."""
    # --- Example 1: Conversation memory (short-term) ---
    # ConversationMemory keeps the full chat history in a sliding window
    memory = ConversationMemory(max_messages=20)
    config = ThundersConfig(model="thunders-7b", temperature=0.7)
    ai = ThundersAI(config=config, memory=memory)

    print("=== Short-term Memory Chat ===")
    # First message — AI learns the user's name
    response = ai.chat("Hi, my name is Alice and I'm a data scientist.")
    print(f"User: Hi, my name is Alice and I'm a data scientist.")
    print(f"AI:   {response}")
    print()

    # Second message — AI should remember the name from context
    response = ai.chat("What career advice would you give me?")
    print(f"User: What career advice would you give me?")
    print(f"AI:   {response}")
    print()

    # Third message — reference earlier context
    response = ai.chat("Can you suggest a learning path for my field?")
    print(f"User: Can you suggest a learning path for my field?")
    print(f"AI:   {response}")
    print()

    # Inspect the conversation memory
    print(f"Messages in memory: {len(memory.get_messages())}")
    for msg in memory.get_messages():
        print(f"  [{msg.role}] {msg.content[:60]}...")
    print()

    # --- Example 2: Long-term memory ---
    # LongTermMemory persists facts across conversations
    long_term = LongTermMemory(storage_path="/tmp/thunders_memory")
    config = ThundersConfig(model="thunders-7b")
    ai = ThundersAI(config=config, memory=long_term)

    print("=== Long-term Memory Chat ===")
    # Store a preference
    response = ai.chat("I prefer dark mode and Python for coding.")
    print(f"User: I prefer dark mode and Python for coding.")
    print(f"AI:   {response}")
    print()

    # Retrieve stored preference in a new topic
    response = ai.chat("What coding environment do I prefer?")
    print(f"User: What coding environment do I prefer?")
    print(f"AI:   {response}")
    print()

    # --- Example 3: Clearing memory ---
    # Reset the conversation to start fresh
    memory.clear()
    print(f"Memory cleared. Messages remaining: {len(memory.get_messages())}")

    response = ai.chat("Do you remember my name?")
    print(f"User: Do you remember my name?")
    print(f"AI:   {response}")


if __name__ == "__main__":
    main()
