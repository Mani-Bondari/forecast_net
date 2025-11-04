import json
from pathlib import Path
from typing import List

from grab_reddit_data import search_reddit_posts
from sentiment_analysis import batch_analyze_sentiment


def combine_text(post: dict) -> str:
    title = post.get("title", "") or ""
    selftext = post.get("selftext", "") or ""
    combined = f"{title} {selftext}".strip()
    return combined if combined else post.get("url", "")


def load_posts_from_file(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        posts = json.load(f)
    print(f"Loaded {len(posts)} posts from {json_path.name}")
    return posts


def main():
    keyword = input("Enter keyword to search on Reddit: ").strip()
    if not keyword:
        print("Keyword is required.")
        return

    print(f"Fetching Reddit posts for keyword '{keyword}'...")
    json_path = search_reddit_posts(keyword, total_limit=1000)
    if json_path is None or not Path(json_path).exists():
        print("Failed to fetch Reddit posts.")
        return

    posts = load_posts_from_file(Path(json_path))

    sentences = []
    for post in posts:
        text = combine_text(post)
        if text:
            sentences.append(text)

    if not sentences:
        print("No text content available for sentiment analysis.")
        return

    results = batch_analyze_sentiment(sentences, batch_size=16)
    avg_polarity = sum(result[5] for result in results) / len(results)

    print(f"Processed {len(results)} posts.")
    print(f"Average adjusted polarity: {avg_polarity:+.4f}")


if __name__ == "__main__":
    main()
