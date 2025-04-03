# Resume Enhancement Bot

A Discord bot that helps users manage and enhance their LaTeX resumes based on job descriptions using AI.

## Features

### Resume Management
- `/setup_resume` - Setup your resume with JSON data and LaTeX template
- `/enhance_resume` - Enhance your resume based on a job description
- `/list_resumes` - List all your saved resumes
- `/delete_resume` - Delete a specific resume

### AI Integration
- Multiple model support (Llama, Gemini, etc.)
- Job description analysis
- ATS optimization
- Healthcare AI domain specialization
- Contextual achievement restructuring

### Chat Features
- `/ask` - Send prompts to the LLM with model/temperature options
- `/model` - List, set, or get info about available models
- `/setkey` - Set your API key
- `/history` - View or clear conversation history
- `/config` - Configure RAG and safety settings

### Security Features
- Input validation and sanitization
- LaTeX command whitelisting
- File size limits
- Rate limiting (30 requests/minute)
- User permission tiers
- API key management

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with required tokens:
```
DISCORD_BOT_TOKEN=your_discord_token
API_KEY=your_api_key
DEFAULT_MODEL=Llama-3.2-3B-Instruct
```

3. Create required directories:
```bash
mkdir -p templates users
```

4. Start the bot:
```bash
python bot.py
```

## Usage

1. **Setup Your Resume**
   ```
   /setup_resume
   # Attach your JSON data and LaTeX template files
   ```

2. **Enhance Your Resume**
   ```
   /enhance_resume <job_description>
   # The bot will analyze and enhance your resume
   ```

3. **Chat with the Bot**
   ```
   /ask <prompt> [--model llama3] [--temp 0.7]
   # Get AI responses with custom parameters
   ```

4. **Manage Your Settings**
   ```
   /model list|set|info
   /config rag|safety
   /history clear|show
   ```

## Project Structure

```
.
├── bot.py                    # Main bot implementation
├── resume_processor.py       # Resume processing and LaTeX operations
├── templates/               # LaTeX templates
├── users/                   # User-specific data
│   └── {discord_user_id}/
│       ├── resume_data.json # User's resume data
│       ├── resume_template.tex # User's LaTeX template
│       └── versions/        # Version history
├── requirements.txt         # Project dependencies
└── .env                     # Environment variables
```

## Dependencies

- discord.py>=2.0.0
- python-dotenv>=1.0.0
- requests>=2.31.0
- aiohttp>=3.9.0
- cachetools>=5.3.0
- python-dateutil>=2.8.2
- jinja2>=3.1.0
- PyPDF2>=3.0.0
- python-docx>=0.8.11
- jsonschema>=4.19.0

## Current Status

- ✅ Bot is running and connected to Discord
- ✅ API integration is working
- ✅ Basic commands are functional
- ✅ User directory structure is set up
- ✅ Resume enhancement pipeline is operational
- ✅ Rate limiting and caching implemented
- ✅ Error handling and logging in place

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 