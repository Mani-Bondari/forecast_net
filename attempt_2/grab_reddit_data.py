import json
import os
import time
from datetime import datetime
from pathlib import Path

import praw
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("client_id")
CLIENT_SECRET = os.getenv("secret")
USER_AGENT = os.getenv("user")


def search_reddit_posts(keyword: str, total_limit: int = 1000, batch_size: int = 100):
    """
    Search top Reddit posts from the past week until we have total_limit posts,
    keeping only posts that actually contain the keyword in the title or body.
    """
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

    posts = []
    seen_ids = set()
    after = None

    keyword_lower = keyword.lower()
    print(f"🔍 Searching for top posts about '{keyword}' until {total_limit} posts are collected...")

    while len(posts) < total_limit:
        try:
            params = {}
            if after:
                params["after"] = after

            submissions = reddit.subreddit("all").search(
                keyword,
                sort="top",
                time_filter="week",
                limit=batch_size,
                params=params
            )

            batch_count = 0
            last_id = None

            for submission in submissions:
                if submission.id in seen_ids:
                    continue
                seen_ids.add(submission.id)
                last_id = submission.fullname

                # ✅ Post-filter: Only keep posts that mention the keyword in title or text
                combined_text = f"{submission.title or ''} {submission.selftext or ''}".lower()
                if keyword_lower not in combined_text:
                    continue

                post_data = {
                    "id": submission.id,
                    "title": submission.title or "",
                    "selftext": submission.selftext or "",
                    "url": submission.url,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "created_date": datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "subreddit": str(submission.subreddit.display_name),
                    "author": submission.author.name if submission.author else None,
                    "permalink": f"https://www.reddit.com{submission.permalink}",
                }

                posts.append(post_data)
                batch_count += 1

                if len(posts) >= total_limit:
                    break

            if not batch_count:
                print("⚠️ No more results found — ending early.")
                break

            after = last_id
            print(f"📦 Collected {len(posts)} posts so far...")
            time.sleep(1.0)  # Respect Reddit API limits

        except Exception as e:
            print(f"❌ Error during fetch: {e}")
            time.sleep(5)
            continue

    posts = sorted(posts, key=lambda x: x["score"], reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reddit_posts_top_{keyword}_{timestamp}.json"
    output_path = Path(__file__).resolve().parent / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(posts)} verified posts to {output_path.name}")
    return output_path


if __name__ == "__main__":
    kw = input("Enter keyword: ").strip()
    search_reddit_posts(kw, total_limit=1000)
