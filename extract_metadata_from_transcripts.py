#!/usr/bin/env python3
"""
Extract basic episode metadata from transcript files themselves.
This creates a basic metadata file that can be enhanced later.
"""

import json
import re
from pathlib import Path
from datetime import datetime

TRANSCRIPTS_DIR = Path(__file__).parent / "Lenny's Podcast Transcripts Archive"
METADATA_FILE = Path(__file__).parent / "episode_metadata.json"


def extract_guest_description(transcript_content: str, guest_name: str) -> str:
    """Extract guest description from transcript intro."""
    # Look for Lenny's intro that describes the guest
    lines = transcript_content.split('\n')[:50]  # First 50 lines usually have intro
    
    intro_text = '\n'.join(lines)
    
    # Look for patterns like "He's the author of..." or "She's the VP of..."
    # Try to find a sentence that describes the guest
    patterns = [
        r"([A-Z][^.!?]*(?:author|founder|CEO|CTO|VP|director|leader|expert|wrote|built|created)[^.!?]*[.!?])",
        r"([A-Z][^.!?]{20,150}[.!?])",  # Any substantial sentence
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, intro_text, re.IGNORECASE)
        if matches:
            # Return the first substantial match
            for match in matches:
                if len(match) > 30 and guest_name.split()[0].lower() in match.lower():
                    return match.strip()
    
    # Fallback
    return f"Product management expert and guest on Lenny's Podcast"


def extract_date_from_content(content: str) -> str:
    """Try to extract date hints from transcript content."""
    # Look for year mentions in the intro
    year_pattern = r'\b(20\d{2})\b'
    years = re.findall(year_pattern, content[:2000])
    if years:
        # Return the most recent year found
        return max(years)
    return "Unknown"


def create_basic_metadata() -> dict:
    """Create basic metadata from transcript files."""
    metadata = {}
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    
    print(f"Processing {len(transcript_files)} transcripts...")
    
    for i, filepath in enumerate(transcript_files, 1):
        guest_name = filepath.stem
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract basic info
            guest_description = extract_guest_description(content, guest_name)
            date = extract_date_from_content(content)
            
            metadata[guest_name] = {
                "guest": guest_name,
                "episode_title": f"Lenny's Podcast: {guest_name}",
                "date": date,
                "guest_description": guest_description
            }
            
            if i % 50 == 0:
                print(f"  Processed {i}/{len(transcript_files)}...")
                
        except Exception as e:
            print(f"  Error processing {guest_name}: {e}")
            metadata[guest_name] = {
                "guest": guest_name,
                "episode_title": f"Lenny's Podcast: {guest_name}",
                "date": "Unknown",
                "guest_description": "Product management expert"
            }
    
    return metadata


def main():
    """Create basic metadata file."""
    print("=" * 60)
    print("Extracting Basic Episode Metadata from Transcripts")
    print("=" * 60)
    
    # Load existing if present
    existing = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            existing = json.load(f)
        print(f"Found {len(existing)} existing entries")
    
    # Create/update metadata
    metadata = create_basic_metadata()
    
    # Merge with existing (existing takes precedence for enhanced entries)
    for guest, info in existing.items():
        if guest in metadata:
            # Keep existing if it has more complete info
            if info.get("date") != "Unknown" or len(info.get("guest_description", "")) > len(metadata[guest].get("guest_description", "")):
                metadata[guest] = info
    
    # Save
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Metadata saved to {METADATA_FILE}")
    print(f"Total episodes: {len(metadata)}")
    print("=" * 60)
    print("\nNote: This is basic metadata extracted from transcripts.")
    print("Run 'python fetch_episode_metadata.py' to enhance with full episode details.")


if __name__ == "__main__":
    main()
