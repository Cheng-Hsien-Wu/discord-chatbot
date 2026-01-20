"""
llmcord - A Discord bot that connects to Google Gemini LLM.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import os
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from dotenv import load_dotenv
from google import genai
from google.genai import types
import httpx
import yaml

# =============================================================================
# Configuration & Constants
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

load_dotenv()

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500
FALLBACK_RESPONSE = "I couldn't generate a response."


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


# =============================================================================
# Global State
# =============================================================================

config = get_config()
curr_model = config["models"][0]
msg_nodes: dict[int, "MsgNode"] = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "Gemini Bot")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MsgNode:
    """Represents a cached message node in the conversation chain."""
    text: str = ""
    images: list[types.Part] = field(default_factory=list)
    role: Literal["user", "model"] = "model"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# =============================================================================
# Permission Checking
# =============================================================================

def check_permissions(
    msg: discord.Message,
    config: dict[str, Any],
    is_dm: bool,
) -> bool:
    """
    Check if the message author has permission to use the bot.
    Returns True if allowed, False if denied.
    """
    permissions = config["permissions"]
    allow_dms = config.get("allow_dms", True)

    user_is_admin = msg.author.id in permissions["users"]["admin_ids"]

    role_ids = set(role.id for role in getattr(msg.author, "roles", ()))
    channel_ids = set(filter(None, (
        msg.channel.id,
        getattr(msg.channel, "parent_id", None),
        getattr(msg.channel, "category_id", None)
    )))

    allowed_user_ids = permissions["users"]["allowed_ids"]
    blocked_user_ids = permissions["users"]["blocked_ids"]
    allowed_role_ids = permissions["roles"]["allowed_ids"]
    blocked_role_ids = permissions["roles"]["blocked_ids"]
    allowed_channel_ids = permissions["channels"]["allowed_ids"]
    blocked_channel_ids = permissions["channels"]["blocked_ids"]

    # User permission check
    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = (
        user_is_admin
        or allow_all_users
        or msg.author.id in allowed_user_ids
        or any(rid in allowed_role_ids for rid in role_ids)
    )
    is_bad_user = (
        not is_good_user
        or msg.author.id in blocked_user_ids
        or any(rid in blocked_role_ids for rid in role_ids)
    )

    # Channel permission check
    allow_all_channels = not allowed_channel_ids
    is_good_channel = (
        (user_is_admin or allow_dms) if is_dm
        else (allow_all_channels or any(cid in allowed_channel_ids for cid in channel_ids))
    )
    is_bad_channel = (
        not is_good_channel
        or any(cid in blocked_channel_ids for cid in channel_ids)
    )

    return not (is_bad_user or is_bad_channel)


# =============================================================================
# Message Chain Building
# =============================================================================

async def fetch_parent_message(
    curr_msg: discord.Message,
    curr_node: MsgNode,
) -> Optional[discord.Message]:
    """
    Determine and fetch the parent message in the conversation chain.
    Returns the parent message or None if this is the chain start.
    """
    try:
        # Case 1: Auto-chain consecutive messages from same author
        if (
            curr_msg.reference is None
            and discord_bot.user.mention not in curr_msg.content
        ):
            prev_msgs = [m async for m in curr_msg.channel.history(before=curr_msg, limit=1)]
            prev_msg = prev_msgs[0] if prev_msgs else None

            if (
                prev_msg
                and prev_msg.type in (discord.MessageType.default, discord.MessageType.reply)
                and prev_msg.author == (
                    discord_bot.user if curr_msg.channel.type == discord.ChannelType.private
                    else curr_msg.author
                )
            ):
                return prev_msg

        # Case 2: Thread start message
        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
        parent_is_thread_start = (
            is_public_thread
            and curr_msg.reference is None
            and curr_msg.channel.parent.type == discord.ChannelType.text
        )

        if parent_is_thread_start:
            return (
                curr_msg.channel.starter_message
                or await curr_msg.channel.parent.fetch_message(curr_msg.channel.id)
            )

        # Case 3: Reply reference
        if curr_msg.reference and curr_msg.reference.message_id:
            return (
                curr_msg.reference.cached_message
                or await curr_msg.channel.fetch_message(curr_msg.reference.message_id)
            )

    except (discord.NotFound, discord.HTTPException):
        logging.exception("Error fetching parent message")
        curr_node.fetch_parent_failed = True

    return None


async def process_message_node(
    curr_msg: discord.Message,
    curr_node: MsgNode,
) -> None:
    """Process a single message and populate its MsgNode data."""
    cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

    good_attachments = [
        att for att in curr_msg.attachments
        if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))
    ]

    attachment_responses = await asyncio.gather(
        *[httpx_client.get(att.url) for att in good_attachments]
    )

    curr_node.text = "\n".join(
        ([cleaned_content] if cleaned_content else [])
        + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text)))
           for embed in curr_msg.embeds]
        + [component.content for component in curr_msg.components
           if component.type == discord.ComponentType.text_display]
        + [resp.text for att, resp in zip(good_attachments, attachment_responses)
           if att.content_type.startswith("text")]
    )

    curr_node.images = [
        types.Part.from_bytes(data=resp.content, mime_type=att.content_type)
        for att, resp in zip(good_attachments, attachment_responses)
        if att.content_type.startswith("image")
    ]

    curr_node.role = "model" if curr_msg.author == discord_bot.user else "user"
    curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
    curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

    curr_node.parent_msg = await fetch_parent_message(curr_msg, curr_node)


async def build_message_chain(
    new_msg: discord.Message,
    max_messages: int,
    max_text: int,
    max_images: int,
) -> tuple[list[types.Content], set[str]]:
    """
    Build the conversation chain from the new message backwards.
    Returns (messages_for_gemini, user_warnings).
    """
    messages: list[types.Content] = []
    user_warnings: set[str] = set()
    curr_msg: Optional[discord.Message] = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == "":
                await process_message_node(curr_msg, curr_node)

            # Format content for Gemini
            parts: list[types.Part] = []
            text_content = curr_node.text[:max_text] if curr_node.text else ""
            if text_content:
                parts.append(types.Part.from_text(text=text_content))
            parts.extend(curr_node.images[:max_images])

            if parts:
                messages.append(types.Content(role=curr_node.role, parts=parts))

            # Collect warnings
            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(
                    f"⚠️ Max {max_images} image{'s' if max_images != 1 else ''} per message"
                    if max_images > 0 else "⚠️ Can't see images"
                )
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'s' if len(messages) != 1 else ''}")

            curr_msg = curr_node.parent_msg

    return messages, user_warnings


# =============================================================================
# System Prompt Processing
# =============================================================================

def process_system_prompt(config: dict[str, Any]) -> Optional[str]:
    """Process and return the system prompt with date/time substitutions."""
    system_prompt = config.get("system_prompt")
    if not system_prompt:
        return None

    now = datetime.now(ZoneInfo("Asia/Taipei"))
    return (
        system_prompt
        .replace("{date}", now.strftime("%B %d %Y"))
        .replace("{time}", now.strftime("%H:%M:%S %Z%z"))
        .strip()
    )


# =============================================================================
# Response Generation
# =============================================================================

async def generate_and_send_response(
    new_msg: discord.Message,
    messages: list[types.Content],
    user_warnings: set[str],
    config: dict[str, Any],
) -> None:
    """Generate LLM response and send it as Discord message(s)."""
    global last_task_time, curr_model

    use_plain_responses = config.get("use_plain_responses", False)
    system_instruction = process_system_prompt(config)

    # Prepare keys and models for fallback loop
    api_keys = [k.strip() for k in os.environ.get("GEMINI_API_KEY", "").split(",") if k.strip()]
    if not api_keys:
        logging.error("No API keys found in GEMINI_API_KEY environment variable")
        return

    models = [curr_model] + [m for m in config["models"] if m != curr_model]
    
    response_msgs: list[discord.Message] = []
    response_contents: list[str] = []

    if use_plain_responses:
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict({
            "fields": [{"name": warning, "value": "", "inline": False} for warning in sorted(user_warnings)]
        })

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)
        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            # Tiered Fallback Loop: Iterate Models -> Iterate Keys
            success = False
            for model in models:
                if success: break
                
                for api_key in api_keys:
                    try:
                        # Create client for this specific key
                        client = genai.Client(api_key=api_key)
                        
                        logging.info(f"Attempting generation with model='{model}' and key='...{api_key[-4:]}'")

                        async for chunk in await client.aio.models.generate_content_stream(
                            model=model,
                            contents=messages[::-1],
                            config=types.GenerateContentConfig(
                                temperature=config.get("temperature", 1.0),
                                system_instruction=system_instruction,
                                tools=[types.Tool(google_search=types.GoogleSearch())] if config.get("use_google_search", True) else None
                            )
                        ):
                            curr_content = chunk.text or ""

                            if not response_contents and curr_content == "":
                                continue

                            start_next_msg = not response_contents or len(response_contents[-1] + curr_content) > max_message_length
                            if start_next_msg:
                                response_contents.append("")

                            response_contents[-1] += curr_content

                            if not use_plain_responses:
                                time_delta = datetime.now().timestamp() - last_task_time
                                ready_to_edit = time_delta >= EDIT_DELAY_SECONDS

                                if start_next_msg or ready_to_edit:
                                    embed.description = response_contents[-1] + STREAMING_INDICATOR
                                    embed.color = EMBED_COLOR_INCOMPLETE

                                    if start_next_msg:
                                        await reply_helper(embed=embed, silent=True)
                                    else:
                                        await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                                        await response_msgs[-1].edit(embed=embed)

                                    last_task_time = datetime.now().timestamp()
                        
                        # If we reached here without exception, success!
                        success = True
                        break

                    except Exception as e:
                        # Log warning but continue to next key/model
                        logging.warning(f"Generation failed with model='{model}' key='...{api_key[-4:]}': {e}")
                        continue
            
            if not success:
                logging.error("All models and keys failed to generate a response.")
                raise Exception("All fallback attempts failed")

            # Final message handling
            if use_plain_responses:
                if response_contents:
                    for content in response_contents:
                        await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))
                else:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=FALLBACK_RESPONSE)))
            else:
                if response_msgs:
                    embed.description = response_contents[-1] if response_contents else FALLBACK_RESPONSE
                    embed.color = EMBED_COLOR_COMPLETE
                    await response_msgs[-1].edit(embed=embed)
                else:
                    embed.description = FALLBACK_RESPONSE
                    embed.color = EMBED_COLOR_INCOMPLETE
                    await reply_helper(embed=embed, silent=True)

    except Exception:
        logging.exception("Error while generating response (all fallbacks exhausted)")

    # Release locks and store response text
    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()


async def cleanup_old_nodes() -> None:
    """Remove oldest message nodes when cache exceeds limit."""
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


# =============================================================================
# Discord Event Handlers
# =============================================================================

@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    """Handle /model command to view or switch the current model."""
    global curr_model

    # Reload config to get latest admin_ids
    current_config = await asyncio.to_thread(get_config)

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        user_is_admin = interaction.user.id in current_config["permissions"]["users"]["admin_ids"]
        if user_is_admin:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(
        output,
        ephemeral=(interaction.channel.type == discord.ChannelType.private)
    )


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    """Provide autocomplete suggestions for /model command."""
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = []
    if curr_str.lower() in curr_model.lower():
        choices.append(Choice(name=f"◉ {curr_model} (current)", value=curr_model))

    choices += [
        Choice(name=f"○ {model}", value=model)
        for model in config["models"]
        if model != curr_model and curr_str.lower() in model.lower()
    ]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    """Handle bot ready event."""
    if client_id := config.get("client_id"):
        logging.info(
            f"\n\nBOT INVITE URL:\n"
            f"https://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n"
        )
    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    """Handle incoming messages."""
    is_dm = new_msg.channel.type == discord.ChannelType.private

    # Ignore messages that don't mention the bot (in non-DM) or are from bots
    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    # Reload config for hot-reload support
    current_config = await asyncio.to_thread(get_config)

    # Check permissions
    if not check_permissions(new_msg, current_config, is_dm):
        return

    max_text = current_config.get("max_text", 100000)
    max_images = current_config.get("max_images", 5)
    max_messages = current_config.get("max_messages", 25)

    # Build message chain
    messages, user_warnings = await build_message_chain(new_msg, max_messages, max_text, max_images)

    logging.info(
        f"Message received (user ID: {new_msg.author.id}, "
        f"attachments: {len(new_msg.attachments)}, "
        f"conversation length: {len(messages)}):\n{new_msg.content}"
    )

    # Generate and send response
    await generate_and_send_response(new_msg, messages, user_warnings, current_config)

    # Cleanup old nodes
    await cleanup_old_nodes()


# =============================================================================
# Main Entry Point
# =============================================================================

async def main() -> None:
    """Main entry point for the bot."""
    try:
        await discord_bot.start(os.environ["DISCORD_BOT_TOKEN"])
    finally:
        await httpx_client.aclose()
        logging.info("Bot shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Received shutdown signal.")
