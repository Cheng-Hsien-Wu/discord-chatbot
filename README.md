# Discord Chatbot (Gemini-Native)

A Discord chatbot powered by Google's Gemini API with native Google Search integration.

## Features

-  **Google Search**: Real-time information via Google Search grounding
-  **Vision**: Analyze images attached to messages
-  **Conversation Memory**: Maintains context across messages (up to 500 cached)
-  **API Fallback**: Multi-key and multi-model fallback for reliability

## Quick Start

### 1. Prerequisites
- Python 3.12+
- Discord Bot Token ([Discord Developer Portal](https://discord.com/developers/applications))
- Gemini API Key ([Google AI Studio](https://aistudio.google.com/apikey))

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/Cheng-Hsien-Wu/discord-chatbot.git
cd discord-chatbot

# Create conda environment
conda create -n discord-bot python=3.12
conda activate discord-bot

# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp config-example.yaml config.yaml
```

### 3. Configuration

Create a `.env` file:
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

## Systemd Deployment (Linux/VPS)

Recommended for Oracle Always Free or small VPS where saving RAM (vs Docker) matters.

1. **Setup Directory**
   Follow the [Quick Start](#quick-start) to set up the environment and config in `/home/opc/discord-chatbot` (or your user path).

2. **Create Service File**
   ```bash
   sudo nano /etc/systemd/system/discord-bot.service
   ```
   Paste the following (adjust `User` and paths as needed):
   ```ini
   [Unit]
   Description=Gemini Discord Bot
   After=network.target

   [Service]
   User=opc
   WorkingDirectory=/home/opc/discord-chatbot
   EnvironmentFile=/home/opc/discord-chatbot/.env
   # Adjust path to your python executable (venv or conda)
   ExecStart=/home/opc/discord-chatbot/venv/bin/python main.py
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

3. **Enable & Start**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable discord-bot
   sudo systemctl start discord-bot
   ```

## Docker Deployment

```bash
docker compose up -d --build
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `max_messages` | Conversation history length | 25 |
| `max_images` | Max images per message | 5 |
| `temperature` | Response creativity (0.0-2.0) | 1.0 |
| `use_google_search` | Enable Google Search | true |

## License

MIT License - See [LICENSE.md](LICENSE.md)

## Acknowledgments

Inspired by [llmcord](https://github.com/jakobdylanc/llmcord) by Jakob Dylan C.
