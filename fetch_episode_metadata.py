#!/usr/bin/env python3
"""
Fetch episode metadata (title, date, guest description) for Lenny's Podcast episodes.
Uses web search to find episode information for each guest.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()

TRANSCRIPTS_DIR = Path(__file__).parent / "Lenny's Podcast Transcripts Archive"
METADATA_FILE = Path(__file__).parent / "episode_metadata.json"


def get_guest_names() -> list[str]:
    """Get all guest names from transcript filenames."""
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    guest_names = [f.stem for f in transcript_files]
    return sorted(guest_names)


def extract_intro_from_transcript(guest_name: str) -> str:
    """Extract the intro section from transcript that describes the guest."""
    transcript_path = TRANSCRIPTS_DIR / f"{guest_name}.txt"
    if not transcript_path.exists():
        return ""
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[:30]  # First 30 lines usually contain intro
    
    intro_text = "".join(lines)
    return intro_text


def fetch_episode_info(guest_name: str, llm: ChatOpenAI) -> dict:
    """Fetch episode metadata for a guest using GPT with transcript intro."""
    
    intro_text = extract_intro_from_transcript(guest_name)
    
    prompt = f"""I need information about a Lenny's Podcast episode featuring {guest_name}.

Here's the intro from the transcript:
{intro_text[:1000]}

Based on this and your knowledge of Lenny's Podcast, please provide:
1. The full episode title (as it appears on the podcast)
2. The publication date (format: YYYY-MM-DD if available, or at least YYYY-MM)
3. A brief 1-2 sentence description of who {guest_name} is (their role, company, expertise)

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
    "guest": "{guest_name}",
    "episode_title": "Full episode title here",
    "date": "YYYY-MM-DD or YYYY-MM",
    "guest_description": "Brief description here"
}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Clean up response to extract JSON
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end > json_start:
                content = content[json_start:json_end].strip()
        
        # Remove any leading/trailing non-JSON text
        if content.startswith("{"):
            json_end = content.rfind("}") + 1
            content = content[:json_end]
        
        metadata = json.loads(content)
        # Ensure all required fields exist
        metadata.setdefault("guest", guest_name)
        metadata.setdefault("episode_title", f"Lenny's Podcast: {guest_name}")
        metadata.setdefault("date", "Unknown")
        metadata.setdefault("guest_description", "Product management expert")
        
        return metadata
    except Exception as e:
        print(f"  Error fetching metadata for {guest_name}: {e}")
        return {
            "guest": guest_name,
            "episode_title": f"Lenny's Podcast: {guest_name}",
            "date": "Unknown",
            "guest_description": "Product management expert"
        }


def main():
    """Fetch metadata for all episodes."""
    print("=" * 60)
    print("Fetching Episode Metadata")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    # Load existing metadata if it exists
    existing_metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            existing_metadata = json.load(f)
        print(f"Loaded {len(existing_metadata)} existing entries")
    
    guest_names = get_guest_names()
    print(f"Found {len(guest_names)} guests")
    
    metadata = {}
    for i, guest_name in enumerate(guest_names, 1):
        if guest_name in existing_metadata:
            print(f"[{i}/{len(guest_names)}] Skipping {guest_name} (already exists)")
            metadata[guest_name] = existing_metadata[guest_name]
        else:
            print(f"[{i}/{len(guest_names)}] Fetching metadata for {guest_name}...")
            metadata[guest_name] = fetch_episode_info(guest_name, llm)
            
            # Save progress after each guest
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Metadata saved to {METADATA_FILE}")
    print(f"Total episodes: {len(metadata)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
