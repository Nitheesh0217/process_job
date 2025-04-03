import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import io
import requests
import json
import time
from typing import Optional, Dict, List, Union
from collections import defaultdict
import re
from datetime import datetime, timedelta
import tempfile
import shutil
import aiohttp
from cachetools import TTLCache
from dateutil import parser
import logging
import sys
from resume_processor import ResumeProcessor

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
API_KEY = os.getenv('API_KEY')

# Check if the tokens are loaded
if TOKEN is None:
    logger.error("DISCORD_BOT_TOKEN not found in .env file.")
    exit()

if API_KEY is None:
    logger.error("API_KEY not found in .env file.")
    exit()

# Define the intents
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent

# Create bot instance with command prefix
bot = commands.Bot(command_prefix='/', intents=intents)

# Store user API keys (in memory for now, should be moved to secure storage)
user_api_keys = {}

# Set default API key for all users
def set_default_api_key():
    """
    Set the default API key for all users from environment variables.
    """
    global user_api_keys
    user_api_keys = defaultdict(lambda: API_KEY)
    logger.info(f"Default API key set: {API_KEY[:10]}...")

# Call set_default_api_key when bot starts
@bot.event
async def on_ready():
    logger.info(f'Bot logged in as {bot.user}')
    set_default_api_key()  # Set default API key for all users
    logger.info("Fetching available models...")
    models = get_available_models()
    if models:
        logger.info("Available models:")
        for model in models:
            logger.info(f"- {model['id']}")
    else:
        logger.warning("No models available or error occurred while fetching models.")

# Response cache (TTL of 1 hour)
response_cache = TTLCache(maxsize=1000, ttl=3600)

# Conversation history (TTL of 24 hours)
conversation_history = TTLCache(maxsize=1000, ttl=86400)

# User configurations
user_configs = defaultdict(lambda: {
    'model': 'Llama-3.2-3B-Instruct',
    'temperature': 0.7,
    'voice': 'alloy',
    'speed': 1.0,
    'rag_enabled': True,
    'safety_level': 'medium'
})

# Rate limiting configuration
RATE_LIMIT = {
    'requests_per_minute': 30,  # Maximum requests per minute
    'window_seconds': 60        # Time window in seconds
}

# Rate limiting tracking
rate_limit_tracker = defaultdict(lambda: {'count': 0, 'window_start': datetime.now()})

# User permission tiers
PERMISSION_TIERS = {
    'admin': 3,
    'moderator': 2,
    'user': 1
}

# Store user permissions (should be moved to database in production)
user_permissions = {}

# Supported file types for RAG processing
SUPPORTED_FILE_TYPES = {
    '.txt': 'text/plain',
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

# Add at the top with other global variables
processed_messages = set()

# Initialize ResumeProcessor
resume_processor = ResumeProcessor()

def format_response(response: str) -> List[str]:
    """
    Format response into Discord-friendly chunks with code blocks when appropriate.
    """
    # Discord's message length limit is 2000 characters
    MAX_LENGTH = 1900  # Leave some room for formatting
    
    # If response is short enough, return as is
    if len(response) <= MAX_LENGTH:
        return [response]
    
    # Split response into paragraphs
    paragraphs = response.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph contains code-like content
        if re.search(r'[{}[\],;]', paragraph) or re.search(r'\b(function|class|def|import|return)\b', paragraph):
            # If current chunk has content, add it
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Add code block
            chunks.append(f"```\n{paragraph}\n```")
        else:
            # Regular text
            if len(current_chunk) + len(paragraph) > MAX_LENGTH:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                current_chunk += paragraph + "\n\n"
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # If we have multiple chunks, add page numbers
    if len(chunks) > 1:
        for i, chunk in enumerate(chunks, 1):
            chunks[i-1] = f"{chunk}\n\n*Page {i}/{len(chunks)}*"
    
    return chunks

def is_rate_limited(user_id: int) -> bool:
    """
    Check if a user has exceeded the rate limit.
    """
    now = datetime.now()
    user_data = rate_limit_tracker[user_id]
    
    # Reset if window has passed
    if (now - user_data['window_start']).total_seconds() > RATE_LIMIT['window_seconds']:
        user_data['count'] = 0
        user_data['window_start'] = now
        return False
    
    # Check if limit exceeded
    if user_data['count'] >= RATE_LIMIT['requests_per_minute']:
        return True
    
    user_data['count'] += 1
    return False

def get_user_permission_level(user_id: int) -> int:
    """
    Get the permission level for a user.
    """
    return user_permissions.get(user_id, PERMISSION_TIERS['user'])

def validate_input(text: str) -> bool:
    """
    Validate and sanitize user input.
    """
    # Remove any potentially harmful characters
    sanitized = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Check length limits
    if len(sanitized) > 4000:  # Discord message limit
        return False
    
    # Check for common injection patterns
    if re.search(r'(?i)(select|insert|update|delete|drop|union|exec|eval)', sanitized):
        return False
    
    return True

def get_available_models():
    """
    Get list of available models from the API.
    """
    url = "https://chat.hpc.fau.edu/openai/models"  # Changed back to original endpoint
    
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        logger.info(f"Making request to {url}")
        logger.info(f"Using API key: {os.getenv('API_KEY')[:10]}...")
        
        response = requests.get(url, headers=headers)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        logger.info(f"Response text: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        else:
            logger.error(f"Error getting models. Status: {response.status_code}")
            logger.error(f"Response text: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return []

def create_chat_payload(model, prompt, user_id=None):
    """
    Create the chat completion payload in the format required by the API.
    """
    messages = []
    
    # Add conversation history if available
    if user_id and user_id in conversation_history:
        # Get last 5 messages for context
        history = conversation_history[user_id][-5:]
        for msg in history:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
    
    # Add current prompt
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    return {
        "model": model,
        "messages": messages
    }

def send_chat_request(prompt, model, api_key, user_id=None):
    """
    Send a chat completion request to the API.
    """
    url = "https://chat.hpc.fau.edu/openai/chat/completions"  # Changed back to original endpoint
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    payload = create_chat_payload(model, prompt, user_id)
    
    try:
        logger.info(f"Making request to {url}")
        logger.info(f"Using API key: {api_key[:10]}...")
        logger.info(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, headers=headers, json=payload)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        logger.info(f"Response text: {response.text}")
        logger.debug(f"Raw API Response: {response.text}")  # Added debug logging for raw response
        
        if response.status_code == 200:
            # Add validation for empty response
            if not response.text.strip():
                logger.error("Empty API response")
                return False, "API returned empty response"
            
            try:
                result = response.json()
                return True, result["choices"][0]["message"]["content"]
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON: {response.text}")
                return False, "API returned malformed JSON"
        else:
            return False, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Request failed: {str(e)}"

async def process_file(file_data: bytes, filename: str) -> str:
    """
    Process uploaded file for RAG processing.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        temp_file.write(file_data)
        temp_path = temp_file.name
    
    try:
        # Process file based on type
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.txt':
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_ext == '.pdf':
            # Add PDF processing logic here
            content = "PDF processing not implemented yet"
        else:
            content = f"Unsupported file type: {file_ext}"
        
        return content
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error deleting temporary file: {e}")

@bot.command(name='setkey')
async def set_api_key(ctx, api_key: str):
    """
    Set the API key for the current user.
    """
    # Check user permission
    if get_user_permission_level(ctx.author.id) < PERMISSION_TIERS['user']:
        await ctx.send("❌ You don't have permission to use this command.")
        return

    # Validate API key format (accept both JWT and sk- format)
    if not (api_key.startswith('sk-') or api_key.startswith('eyJ')):
        await ctx.send("❌ Invalid API key format. API key should start with 'sk-' or be a JWT token.")
        return
    
    # Store the API key (should be encrypted in production)
    user_api_keys[ctx.author.id] = api_key
    await ctx.send("✅ API key set successfully!")

@bot.command(name='model')
async def model_command(ctx, action: str, model_name: Optional[str] = None):
    """
    Handle model-related commands.
    Usage: /model list|set|info
    """
    if action == 'list':
        models = get_available_models()
        if models:
            model_list = "\n".join([f"- {model['id']}" for model in models])
            await ctx.send(f"Available models:\n{model_list}")
        else:
            await ctx.send("No models available or error occurred while fetching models.")
    
    elif action == 'set':
        if not model_name:
            await ctx.send("Please specify a model name")
            return
        
        models = get_available_models()
        model_ids = [model['id'] for model in models]
        
        if model_name in model_ids:
            user_configs[ctx.author.id]['model'] = model_name
            await ctx.send(f"✅ Model set to: {model_name}")
        else:
            await ctx.send(f"❌ Invalid model name. Use '/model list' to see available models.")
    
    elif action == 'info':
        current_model = user_configs[ctx.author.id]['model']
        await ctx.send(f"Current model: {current_model}")
    
    else:
        await ctx.send("Invalid action. Use 'list', 'set', or 'info'")

@bot.command(name='setrole')
@commands.has_permissions(administrator=True)
async def set_role(ctx, user: discord.Member, role: str):
    """
    Set a user's permission role (admin only).
    """
    if role.lower() not in PERMISSION_TIERS:
        await ctx.send(f"❌ Invalid role. Available roles: {', '.join(PERMISSION_TIERS.keys())}")
        return
    
    user_permissions[user.id] = PERMISSION_TIERS[role.lower()]
    await ctx.send(f"✅ Set {user.name}'s role to {role}")

@bot.tree.command(name="explain", description="Explain the selected message using OwlChat")
async def explain_message(interaction: discord.Interaction):
    """
    Context menu command to explain a message using OwlChat.
    """
    # Get the selected message
    message = interaction.data.get('target_id')
    if not message:
        await interaction.response.send_message("❌ No message selected.", ephemeral=True)
        return

    # Check user permission
    if get_user_permission_level(interaction.user.id) < PERMISSION_TIERS['user']:
        await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
        return

    # Check rate limit
    if is_rate_limited(interaction.user.id):
        await interaction.response.send_message("⚠️ You've reached the rate limit. Please wait a minute before trying again.", ephemeral=True)
        return

    # Get user's API key
    api_key = user_api_keys.get(interaction.user.id)
    if not api_key:
        await interaction.response.send_message("Please set your API key first using the /setkey command", ephemeral=True)
        return

    # Get user's selected model
    model = user_configs[interaction.user.id]['model']
    
    # Create prompt
    prompt = f"Please explain this message: {message.content}"
    
    await interaction.response.defer()
    
    success, response = send_chat_request(prompt, model, api_key, interaction.user.id)
    
    if success:
        chunks = format_response(response)
        for chunk in chunks:
            await interaction.followup.send(chunk)
    else:
        await interaction.followup.send(f"❌ Error: {response}")

@bot.tree.command(name="summarize", description="Summarize the selected message using OwlChat")
async def summarize_message(interaction: discord.Interaction):
    """
    Context menu command to summarize a message using OwlChat.
    """
    # Get the selected message
    message = interaction.data.get('target_id')
    if not message:
        await interaction.response.send_message("❌ No message selected.", ephemeral=True)
        return

    # Check user permission
    if get_user_permission_level(interaction.user.id) < PERMISSION_TIERS['user']:
        await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
        return

    # Check rate limit
    if is_rate_limited(interaction.user.id):
        await interaction.response.send_message("⚠️ You've reached the rate limit. Please wait a minute before trying again.", ephemeral=True)
        return

    # Get user's API key
    api_key = user_api_keys.get(interaction.user.id)
    if not api_key:
        await interaction.response.send_message("Please set your API key first using the /setkey command", ephemeral=True)
        return

    # Get user's selected model
    model = user_configs[interaction.user.id]['model']
    
    # Create prompt
    prompt = f"Please provide a concise summary of this message: {message.content}"
    
    await interaction.response.defer()
    
    success, response = send_chat_request(prompt, model, api_key, interaction.user.id)
    
    if success:
        chunks = format_response(response)
        for chunk in chunks:
            await interaction.followup.send(chunk)
    else:
        await interaction.followup.send(f"❌ Error: {response}")

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if we've already processed this message
    if message.id in processed_messages:
        return

    # Add message to processed set
    processed_messages.add(message.id)

    # Process commands first
    await bot.process_commands(message)

    # Check if the message has attachments
    if message.attachments:
        for attachment in message.attachments:
            file_ext = os.path.splitext(attachment.filename)[1].lower()
            
            # Handle PDF files
            if file_ext == '.pdf':
                try:
                    pdf_data = await attachment.read()
                    discord_file = discord.File(fp=io.BytesIO(pdf_data), filename=attachment.filename)
                    await message.reply(f"I got this PDF file: {attachment.filename}", file=discord_file)
                    return
                except Exception as e:
                    logger.error(f"Error handling PDF attachment: {e}")
                    await message.reply("Sorry, I had trouble processing that PDF.")
                    return
            
            # Handle RAG processing for supported file types
            elif file_ext in SUPPORTED_FILE_TYPES:
                try:
                    file_data = await attachment.read()
                    content = await process_file(file_data, attachment.filename)
                    
                    # Get user's API key and model
                    api_key = user_api_keys.get(message.author.id, API_KEY)
                    if not api_key:
                        await message.reply("Please set your API key first using the /setkey command")
                        return
                    
                    model = user_configs[message.author.id]['model']
                    
                    # Create prompt for RAG
                    prompt = f"Please analyze this content and provide insights:\n\n{content}"
                    
                    # Update conversation history
                    if message.author.id not in conversation_history:
                        conversation_history[message.author.id] = []
                    conversation_history[message.author.id].append({
                        'role': 'user',
                        'content': prompt,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    success, response = send_chat_request(prompt, model, api_key, message.author.id)
                    
                    if success:
                        # Update conversation history with response
                        conversation_history[message.author.id].append({
                            'role': 'assistant',
                            'content': response,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        chunks = format_response(response)
                        for chunk in chunks:
                            await message.reply(chunk)
                    else:
                        await message.reply(f"❌ Error: {response}")
                    
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    await message.reply("Sorry, I had trouble processing that file.")
                    return

    # Handle text messages
    if message.content and not message.content.startswith('/'):
        # Check user permission
        if get_user_permission_level(message.author.id) < PERMISSION_TIERS['user']:
            await message.reply("❌ You don't have permission to use this bot.")
            return

        # Check rate limit
        if is_rate_limited(message.author.id):
            await message.reply("⚠️ You've reached the rate limit. Please wait a minute before trying again.")
            return

        # Validate input
        if not validate_input(message.content):
            await message.reply("❌ Invalid input. Please check your message and try again.")
            return

        # Get user's API key (use default from environment if not set)
        api_key = user_api_keys.get(message.author.id, API_KEY)
        if not api_key:
            await message.reply("Please set your API key first using the /setkey command")
            return

        prompt = message.content
        logger.info(f"\nReceived prompt from {message.author}: {prompt}")

        # Get user's selected model or use default
        model = user_configs[message.author.id]['model']
        logger.info(f"Using model: {model}")
        
        # Update conversation history
        if message.author.id not in conversation_history:
            conversation_history[message.author.id] = []
        conversation_history[message.author.id].append({
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().isoformat()
        })
        
        # Indicate bot is thinking
        async with message.channel.typing():
            success, response = send_chat_request(prompt, model, api_key, message.author.id)
            
            if success:
                # Update conversation history with response
                conversation_history[message.author.id].append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                chunks = format_response(response)
                for chunk in chunks:
                    await message.reply(chunk)
            else:
                logger.error(f"Chat request failed: {response}")
                await message.reply("Sorry, I encountered an error. Please try again later.")

@bot.command(name='ask')
async def ask_command(ctx, *, prompt: str):
    """
    Send a prompt to the LLM with optional parameters.
    Usage: /ask <prompt> [--model llama3] [--temp 0.7]
    """
    # Parse command arguments
    args = prompt.split()
    model = user_configs[ctx.author.id]['model']
    temperature = user_configs[ctx.author.id]['temperature']
    
    # Extract parameters
    prompt_text = prompt
    for i, arg in enumerate(args):
        if arg == '--model' and i + 1 < len(args):
            model = args[i + 1]
            prompt_text = prompt_text.replace(f"--model {model}", "").strip()
        elif arg == '--temp' and i + 1 < len(args):
            try:
                temperature = float(args[i + 1])
                prompt_text = prompt_text.replace(f"--temp {temperature}", "").strip()
            except ValueError:
                await ctx.send("❌ Invalid temperature value. Using default.")
    
    # Check cache
    cache_key = f"{ctx.author.id}:{prompt_text}:{model}:{temperature}"
    if cache_key in response_cache:
        await ctx.send("(Cached response)")
        chunks = format_response(response_cache[cache_key])
        for chunk in chunks:
            await ctx.send(chunk)
        return
    
    # Get user's API key
    api_key = user_api_keys.get(ctx.author.id)
    if not api_key:
        await ctx.send("Please set your API key first using the /apikey set command")
        return
    
    # Update conversation history
    if ctx.author.id not in conversation_history:
        conversation_history[ctx.author.id] = []
    conversation_history[ctx.author.id].append({
        'role': 'user',
        'content': prompt_text,
        'timestamp': datetime.now().isoformat()
    })
    
    # Send request
    success, response = send_chat_request(prompt_text, model, api_key, ctx.author.id)
    
    if success:
        # Cache the response
        response_cache[cache_key] = response
        
        # Update conversation history
        conversation_history[ctx.author.id].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        chunks = format_response(response)
        for chunk in chunks:
            await ctx.send(chunk)
    else:
        await ctx.send(f"❌ Error: {response}")

@bot.command(name='tts')
async def tts_command(ctx, *, text: str):
    """
    Generate text-to-speech from text.
    Usage: /tts <text> [--voice alloy] [--speed 1.2]
    """
    # Parse command arguments
    args = text.split()
    voice = user_configs[ctx.author.id]['voice']
    speed = user_configs[ctx.author.id]['speed']
    
    # Extract parameters
    tts_text = text
    for i, arg in enumerate(args):
        if arg == '--voice' and i + 1 < len(args):
            voice = args[i + 1]
            tts_text = tts_text.replace(f"--voice {voice}", "").strip()
        elif arg == '--speed' and i + 1 < len(args):
            try:
                speed = float(args[i + 1])
                tts_text = tts_text.replace(f"--speed {speed}", "").strip()
            except ValueError:
                await ctx.send("❌ Invalid speed value. Using default.")
    
    # Generate TTS
    audio_data = await generate_tts(tts_text, voice, speed)
    
    if audio_data:
        # Send audio file
        file = discord.File(fp=io.BytesIO(audio_data), filename='response.mp3')
        await ctx.send("Here's your audio:", file=file)
    else:
        await ctx.send("❌ Failed to generate audio.")

@bot.command(name='apikey')
async def apikey_command(ctx, action: str):
    """
    Handle API key commands.
    Usage: /apikey set|reset
    """
    if action == 'set':
        await ctx.send("Please enter your API key:")
        try:
            api_key = await bot.wait_for('message', timeout=30.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
            if api_key.content.startswith('sk-') or api_key.content.startswith('eyJ'):
                user_api_keys[ctx.author.id] = api_key.content
                await ctx.send("✅ API key set successfully!")
            else:
                await ctx.send("❌ Invalid API key format. API key should start with 'sk-' or be a JWT token.")
        except TimeoutError:
            await ctx.send("❌ Timeout. Please try again.")
    
    elif action == 'reset':
        if ctx.author.id in user_api_keys:
            del user_api_keys[ctx.author.id]
            await ctx.send("✅ API key reset successfully!")
        else:
            await ctx.send("No API key found to reset.")
    
    else:
        await ctx.send("Invalid action. Use 'set' or 'reset'")

@bot.command(name='history')
async def history_command(ctx, action: str):
    """
    Handle conversation history commands.
    Usage: /history clear|show
    """
    if action == 'clear':
        if ctx.author.id in conversation_history:
            del conversation_history[ctx.author.id]
            await ctx.send("✅ Conversation history cleared!")
        else:
            await ctx.send("No conversation history to clear.")
    
    elif action == 'show':
        if ctx.author.id in conversation_history:
            history = conversation_history[ctx.author.id]
            if history:
                # Format history for display
                formatted_history = []
                for msg in history[-5:]:  # Show last 5 messages
                    role = "You" if msg['role'] == 'user' else "Assistant"
                    timestamp = parser.parse(msg['timestamp']).strftime("%H:%M:%S")
                    formatted_history.append(f"[{timestamp}] {role}: {msg['content'][:100]}...")
                
                await ctx.send("Recent conversation history:\n" + "\n".join(formatted_history))
            else:
                await ctx.send("No conversation history available.")
        else:
            await ctx.send("No conversation history available.")
    
    else:
        await ctx.send("Invalid action. Use 'clear' or 'show'")

@bot.command(name='config')
async def config_command(ctx, category: str, value: Optional[str] = None):
    """
    Handle configuration commands.
    Usage: /config rag|safety
    """
    if category == 'rag':
        if value is None:
            current = user_configs[ctx.author.id]['rag_enabled']
            await ctx.send(f"RAG is currently {'enabled' if current else 'disabled'}")
        else:
            if value.lower() in ['on', 'enable', 'true']:
                user_configs[ctx.author.id]['rag_enabled'] = True
                await ctx.send("✅ RAG enabled")
            elif value.lower() in ['off', 'disable', 'false']:
                user_configs[ctx.author.id]['rag_enabled'] = False
                await ctx.send("✅ RAG disabled")
            else:
                await ctx.send("❌ Invalid value. Use 'on' or 'off'")
    
    elif category == 'safety':
        if value is None:
            current = user_configs[ctx.author.id]['safety_level']
            await ctx.send(f"Current safety level: {current}")
        else:
            if value.lower() in ['low', 'medium', 'high']:
                user_configs[ctx.author.id]['safety_level'] = value.lower()
                await ctx.send(f"✅ Safety level set to {value.lower()}")
            else:
                await ctx.send("❌ Invalid safety level. Use 'low', 'medium', or 'high'")
    
    else:
        await ctx.send("Invalid category. Use 'rag' or 'safety'")

@bot.command(name='setup_resume')
async def setup_resume(ctx):
    """Setup resume template and data"""
    try:
        if len(ctx.message.attachments) != 2:
            return await ctx.send("❌ Please attach both JSON data and LaTeX template files")
        
        # Validate attachments
        json_attach = next((a for a in ctx.message.attachments if a.filename.endswith('.json')), None)
        tex_attach = next((a for a in ctx.message.attachments if a.filename.endswith('.tex')), None)
        
        if not json_attach or not tex_attach:
            return await ctx.send("❌ Please attach one JSON file and one LaTeX template file")
        
        user_dir = f"users/{ctx.author.id}"
        os.makedirs(user_dir, exist_ok=True)
        
        # Save files
        await json_attach.save(f"{user_dir}/resume_data.json")
        await tex_attach.save(f"{user_dir}/resume_template.tex")
        
        # Validate JSON structure
        with open(f"{user_dir}/resume_data.json") as f:
            resume_data = json.load(f)
            if not resume_processor.validate_resume(resume_data):
                return await ctx.send("❌ Invalid JSON structure. Please check the format.")
        
        # Validate LaTeX template
        with open(f"{user_dir}/resume_template.tex") as f:
            template = f.read()
            if not resume_processor.validate_latex(template):
                return await ctx.send("❌ Invalid LaTeX template. Please check the format.")
        
        await ctx.send("✅ Resume setup complete! Your resume data and template have been saved.")
        
    except Exception as e:
        logger.error(f"Error in setup_resume: {str(e)}")
        await ctx.send(f"❌ An error occurred: {str(e)}")

@bot.command(name='enhance_resume')
async def enhance_resume(ctx, *, job_description: str):
    """Enhance resume with job description"""
    try:
        user_dir = f"users/{ctx.author.id}"
        
        # Validate setup
        if not os.path.exists(f"{user_dir}/resume_data.json"):
            return await ctx.send("❌ Please setup your resume first with /setup_resume")
        
        async with ctx.typing():
            # Load data and template
            with open(f"{user_dir}/resume_data.json") as f:
                resume_data = json.load(f)
            
            with open(f"{user_dir}/resume_template.tex") as f:
                template = f.read()
            
            # Enhance with AI
            enhanced_data = await resume_processor.enhance_with_ai(
                resume_data=resume_data,
                job_description=job_description,
                user_id=ctx.author.id
            )
            
            # Generate LaTeX
            latex_content = resume_processor.render_latex(enhanced_data, template)
            
            # Compile PDF
            pdf_path = resume_processor.compile_latex(latex_content, user_dir)
            
            # Send the enhanced PDF
            await ctx.send(
                "✅ Resume enhanced successfully! Here's your enhanced version:",
                file=discord.File(pdf_path, "enhanced_resume.pdf")
            )
            
    except Exception as e:
        logger.error(f"Error in enhance_resume: {str(e)}")
        await ctx.send(f"❌ An error occurred: {str(e)}")

@bot.command(name='list_resumes')
async def list_resumes(ctx):
    """List all resumes for the user"""
    try:
        user_dir = os.path.join("users", str(ctx.author.id))
        
        if not os.path.exists(user_dir):
            return await ctx.send("❌ No resumes found. Use /setup_resume to create one.")
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(user_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            return await ctx.send("❌ No resumes found. Use /setup_resume to create one.")
        
        # Create a formatted list
        resume_list = "**Your Resumes:**\n"
        for pdf_file in pdf_files:
            file_path = os.path.join(user_dir, pdf_file)
            file_size = os.path.getsize(file_path) / 1024  # Convert to KB
            file_date = datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d %H:%M")
            resume_list += f"\n- {pdf_file} ({file_size:.1f} KB, created {file_date})"
        
        await ctx.send(resume_list)
        
    except Exception as e:
        logger.error(f"Error in list_resumes: {str(e)}")
        await ctx.send(f"❌ Error listing resumes: {str(e)}")

@bot.command(name='delete_resume')
async def delete_resume(ctx, filename: str):
    """Delete a specific resume"""
    try:
        user_dir = os.path.join("users", str(ctx.author.id))
        file_path = os.path.join(user_dir, filename)
        
        if not os.path.exists(file_path):
            return await ctx.send("❌ Resume not found.")
        
        # Ensure the file is a PDF and belongs to the user
        if not filename.endswith('.pdf') or not file_path.startswith(user_dir):
            return await ctx.send("❌ Invalid file.")
        
        os.remove(file_path)
        await ctx.send(f"✅ Deleted resume: {filename}")
        
    except Exception as e:
        logger.error(f"Error in delete_resume: {str(e)}")
        await ctx.send(f"❌ Error deleting resume: {str(e)}")

# Run the bot with the token
bot.run(TOKEN) 