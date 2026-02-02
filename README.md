# Lenny's Podcast Knowledge Base

A RAG-powered Streamlit application that allows product managers to query insights from 302 Lenny's Podcast transcripts using GPT 5.2.

## Features

- 🔍 Semantic search across 302 expert interviews
- 💬 Chat interface with expert attribution
- 📚 Detailed source citations with episode metadata
- 🎯 PM-focused responses for strategic questions and PRD reviews

## Local Development

### Prerequisites

- Python 3.9+
- OpenAI API key

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Extract episode metadata (optional but recommended):**
   ```bash
   python extract_metadata_from_transcripts.py
   ```

4. **Index the transcripts:**
   ```bash
   python ingest.py
   ```
   This will process all 302 transcripts, chunk them, and create embeddings. Takes ~5-10 minutes.

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Deployment

### Step 1: Push to GitHub

1. Initialize git repository (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Create a new repository on GitHub and push:
   ```bash
   git remote add origin https://github.com/yourusername/lenny-podcast-kb.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to `app.py`

### Step 3: Configure Secrets

1. In your Streamlit Cloud app settings, go to "Secrets"
2. Add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

### Step 4: Deploy

1. Click "Deploy"
2. Streamlit Cloud will automatically:
   - Install dependencies from `requirements.txt`
   - Run your app
   - Make it publicly accessible

## Important Notes for Cloud Deployment

- **File Size Limits**: Streamlit Cloud has a 1GB repository limit. The `chroma_db` directory (~50-100MB) and transcripts (~25MB) should fit within this limit.
- **First Run**: The app will use the pre-indexed `chroma_db` directory. If it's missing, you'll need to run `ingest.py` locally and commit the `chroma_db` folder.
- **API Costs**: Each query uses OpenAI API calls for embeddings and GPT-5.2. Monitor your usage.

## Project Structure

```
lenny/
├── app.py                          # Main Streamlit application
├── ingest.py                        # Data ingestion script
├── extract_metadata_from_transcripts.py  # Extract basic metadata
├── fetch_episode_metadata.py        # Enhance metadata with GPT (optional)
├── requirements.txt                 # Python dependencies
├── episode_metadata.json            # Episode metadata
├── chroma_db/                       # Vector database (generated)
├── Lenny's Podcast Transcripts Archive/  # Transcript files
└── recare_logo_rgb_master.png      # Logo image
```

## Troubleshooting

- **"Knowledge base not found"**: Run `python ingest.py` to create the vector database
- **API key errors**: Ensure your API key is set in `.env` (local) or Streamlit secrets (cloud)
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
