#!/usr/bin/env python3
"""
run_example.py
==============
Simple interactive script to run the dialogue frame finder.
Just run this and it will ask you for the URL and dialogue!

Usage:
    python run_example.py
"""

import subprocess
import sys


def main():
    print("=" * 70)
    print("Dialogue Frame Finder - Interactive Runner")
    print("=" * 70)
    print()

    # Get URL from user
    print("Enter the video URL (YouTube URL or local file path):")
    print("Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print("     or: C:\\videos\\my_video.mp4")
    url = input("> ").strip()

    if not url:
        print("Error: URL cannot be empty!")
        sys.exit(1)

    print()

    # Get target dialogue from user
    print("Enter the dialogue to search for:")
    print("Example: my mind rebels at stagnation")
    dialogue = input("> ").strip()

    if not dialogue:
        print("Error: Dialogue cannot be empty!")
        sys.exit(1)

    print()

    # Ask for verbose mode
    print("Enable verbose output? (y/n) [default: n]")
    verbose = input("> ").strip().lower() == 'y'

    print()

    # Ask if they want to skip audio
    print("Skip audio processing (faster, use if dialogue is on-screen text)? (y/n) [default: n]")
    no_audio = input("> ").strip().lower() == 'y'

    print()
    print("=" * 70)
    print("Starting search...")
    print("=" * 70)
    print()

    # Build command
    cmd = [sys.executable, "main.py", "--url", url, "--text", dialogue]

    if verbose:
        cmd.append("--verbose")

    if no_audio:
        cmd.append("--no-audio")

    # Run the main script
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nSearch cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
