# Discord Chatbot (Gemini-Native)

A Discord chatbot powered by Google's Gemini API. Fork of [llmcord](https://github.com/jakobdylanc/llmcord), rewritten to use Gemini's native SDK.

## Features

### Chat System (from llmcord)
- **Reply-based conversations**: @ the bot to start, reply to continue. Build branching conversations.
- **Thread support**: Create threads from any message and continue inside.
- **Auto-chaining**: Back-to-back messages from the same user are automatically linked.
- **DM mode**: In DMs, conversations continue automatically.

### Bot Framework (from llmcord)
- **Model switching**: Use `/model` to switch between models.
- **Multi-model fallback**: If one model fails, try the next.
- **Streaming responses**: See responses in real-time (turns green when complete).
- **Permission system**: Control access by user, role, or channel.
- **Hot-reload config**: Change settings without restarting.

### Gemini Integration (this fork)
- **Google Search**: Real-time web search via Gemini's native grounding. *(Gemini capability, integrated by this project)*
- **Vision**: Analyze images attached to messages. *(Gemini capability, integrated by this project)*

## Quick Start

### 1. Prerequisites
- Python 3.12+
- Discord Bot Token ([Discord Developer Portal](https://discord.com/developers/applications))
- Gemini API Key ([Google AI Studio](https://aistudio.google.com/apikey))

### 2. Setup

```bash
git clone https://github.com/Cheng-Hsien-Wu/discord-chatbot.git
cd discord-chatbot

conda create -n discord-bot python=3.12
conda activate discord-bot
pip install -r requirements.txt

cp config-example.yaml config.yaml
```

### 3. Configuration

Create `.env`:
```env
DISCORD_BOT_TOKEN=your_discord_bot_token
GEMINI_API_KEY=your_gemini_api_key
```

Edit `config.yaml`:
```yaml
client_id: your_bot_client_id
permissions:
  users:
    admin_ids: [your_discord_user_id]
```

### 4. Run

```bash
python main.py
```

Or with Docker:
```bash
docker compose up -d
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `max_messages` | Conversation history length | 25 |
| `max_images` | Max images per message | 5 |
| `temperature` | Response creativity (0.0-2.0) | 1.0 |
| `use_google_search` | Enable Google Search | true |
| `allow_dms` | Allow direct messages | false |

## License

MIT License - See [LICENSE](LICENSE)

## Acknowledgments

This project is a fork of [llmcord](https://github.com/jakobdylanc/llmcord) by Jakob Dylan C.

The following features are inherited from llmcord:
- Reply-based chat system with threading and auto-chaining
- `/model` command and multi-model fallback
- Streaming responses with Discord message editing
- Permission system (user/role/channel)
- Hot-reload configuration
- Message caching with mutex protection

This fork replaces the OpenAI-compatible API layer with Google's native `google-genai` SDK to enable Gemini-specific features like Google Search grounding. The codebase is also reorganized with section headers and clearer function structure for improved readability.
