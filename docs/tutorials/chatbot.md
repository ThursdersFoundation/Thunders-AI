# Building a Chatbot with Thunders AI

In this tutorial, you'll build a fully-featured conversational chatbot using Thunders AI. The chatbot will support multi-turn conversations, persistent memory, streaming responses, and customizable personality through system prompts. By the end, you'll have a production-ready chatbot that can be deployed as a command-line application, web service, or integrated into larger systems.

## What You'll Build

A conversational AI chatbot called **ThunderBot** that:
- Maintains context across multi-turn conversations
- Remembers user preferences using the memory system
- Streams responses in real time
- Can be customized with different personalities
- Handles errors gracefully with fallback responses

## Prerequisites

- Python 3.9+
- Thunders AI installed (`pip install thunders-ai`)

## Step 1: Basic Chatbot

Let's start with the simplest possible chatbot:

```python
from thunders_ai import ThundersAI

ai = ThundersAI()

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit", "bye"):
        print("ThunderBot: Goodbye!")
        break
    response = ai.chat(user_input)
    print(f"ThunderBot: {response}")
```

This works, but it doesn't maintain conversation context between messages. Each call to `ai.chat()` is independent.

## Step 2: Adding Conversation Memory

To build a multi-turn chatbot, we need to track the conversation history. Thunders AI's `ChatHistory` class handles this automatically:

```python
from thunders_ai import ThundersAI, ThundersConfig
from thunders_ai.llm.chat_model import ChatHistory

config = ThundersConfig(model="thunders-7b")
ai = ThundersAI(config=config)
history = ChatHistory(max_length=50)  # Remember last 50 messages

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        break

    # Add user message to history
    history.add_message("user", user_input)

    # Send conversation context to the AI
    response = ai.chat(user_input, history=history)

    # Add assistant response to history
    history.add_message("assistant", response)
    print(f"ThunderBot: {response}")
```

## Step 3: Custom Personality

Use system prompts to give your chatbot a unique personality:

```python
config = ThundersConfig(
    model="thunders-7b",
    system_prompt=(
        "You are ThunderBot, a witty and knowledgeable AI assistant. "
        "You love thunderstorms and often make weather-related puns. "
        "Keep your answers concise but entertaining. "
        "If you don't know something, say so honestly."
    ),
)
ai = ThundersAI(config=config)
```

## Step 4: Long-Term Memory with MemorySystem

For remembering user preferences across sessions, use the persistent `MemorySystem`:

```python
from thunders_ai.llm.memory import MemorySystem, Memory, MemoryType

memory = MemorySystem(storage_path="./thunderbot_memories")

# Save a user preference
memory.save(Memory(
    id="pref-theme",
    content="User prefers dark theme",
    memory_type=MemoryType.PREFERENCE,
    timestamp=time.time(),
))

# Later, search for relevant memories before responding
relevant = memory.search("user preferences")
context = "\n".join([m.content for m in relevant])
response = ai.chat(f"Context: {context}\n\nUser: {user_input}")
```

## Step 5: Streaming Responses

For a more responsive feel, stream tokens as they're generated:

```python
print("ThunderBot: ", end="", flush=True)
for token in ai.chat_stream(user_input):
    print(token, end="", flush=True)
print()  # Newline after streaming completes
```

## Step 6: Complete Chatbot

Here's the full, production-ready chatbot:

```python
"""ThunderBot — A conversational AI chatbot built with Thunders AI."""
import time
from thunders_ai import ThundersAI, ThundersConfig
from thunders_ai.llm.chat_model import ChatHistory
from thunders_ai.llm.memory import MemorySystem, Memory, MemoryType

def main():
    config = ThundersConfig(
        model="thunders-7b",
        system_prompt=(
            "You are ThunderBot, a witty and knowledgeable AI assistant. "
            "You love thunderstorms and often make weather-related puns. "
            "Keep answers concise but entertaining."
        ),
    )
    ai = ThundersAI(config=config)
    history = ChatHistory(max_length=50)
    memory = MemorySystem(storage_path="./thunderbot_memories")

    print("ThunderBot: Hello! I'm ThunderBot. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThunderBot: Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("ThunderBot: Goodbye! ⚡")
            break

        # Search for relevant memories
        relevant = memory.search(user_input)
        memory_context = "\n".join([m.content for m in relevant[:5]])

        # Stream the response
        print("ThunderBot: ", end="", flush=True)
        full_response = ""
        for token in ai.chat_stream(user_input, history=history):
            print(token, end="", flush=True)
            full_response += token
        print()

        # Update history
        history.add_message("user", user_input)
        history.add_message("assistant", full_response)

if __name__ == "__main__":
    main()
```

## Next Steps

- Deploy this chatbot as a **web service** using the API server (see [Quick Start](../quickstart.md))
- Add **vision capabilities** so the chatbot can discuss images
- Implement **function calling** to let the chatbot take actions (e.g., looking up weather)
- Add **authentication** and **rate limiting** for public-facing deployments
