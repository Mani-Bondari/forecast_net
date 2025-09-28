import os
import praw
from datetime import datetime
from dotenv import load_dotenv
import argparse
from psaw import PushshiftAPI

load_dotenv()

# Initialize Reddit client (PRAW)
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT"),
    client_secret=os.getenv("SECRET_KEY"),
    user_agent="forecast_net/0.1 by u/LookTurbulent426"
)

# Initialize Pushshift API (for historical)
psaw_api = PushshiftAPI()

def fetch_reddit_posts(keyword: str, start_date: str, end_date: str, limit: int = 2000, subreddit: str = "all"):
    """
    Fetch Reddit posts by keyword and date range.
    Uses PRAW if recent (last 30 days), PSAW otherwise.
    """
    start_epoch = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_epoch = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    now_epoch = int(datetime.utcnow().timestamp())
    days_old = (now_epoch - end_epoch) / 86400  # days since end_date

    posts = []

    # -------------------------------
    # Case 1: Recent data → use PRAW
    # -------------------------------
    if days_old <= 30:
        print("⚡ Using PRAW for recent data")
        total_fetched = 0
        for submission in reddit.subreddit(subreddit).new(limit=None):
            created = int(submission.created_utc)

            if created < start_epoch:
                break  # stop once past start date

            if start_epoch <= created <= end_epoch and keyword.lower() in submission.title.lower():
                author_karma = None
                if submission.author:
                    try:
                        author_karma = submission.author.link_karma + submission.author.comment_karma
                    except Exception:
                        pass

                posts.append({
                    "title": submission.title,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "author": str(submission.author),
                    "author_karma": author_karma,
                    "url": submission.url,
                    "created_utc": datetime.utcfromtimestamp(created).isoformat()
                })
                total_fetched += 1
                if total_fetched >= limit:
                    break

    # -------------------------------
    # Case 2: Older data → use PSAW
    # -------------------------------
    else:
        print("📜 Using Pushshift (PSAW) for historical data")
        submissions = psaw_api.search_submissions(
            q=keyword,
            after=start_epoch,
            before=end_epoch,
            subreddit=subreddit,
            limit=limit
        )

        for submission in submissions:
            author_karma = None
            if submission.author:
                try:
                    redditor = reddit.redditor(str(submission.author))
                    author_karma = redditor.link_karma + redditor.comment_karma
                except Exception:
                    pass

            posts.append({
                "title": submission.title,
                "score": getattr(submission, "score", None),
                "num_comments": getattr(submission, "num_comments", None),
                "author": str(submission.author),
                "author_karma": author_karma,
                "url": submission.url,
                "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat()
            })

    return posts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Reddit posts by keyword and date range.")
    parser.add_argument("keyword", type=str, help="Search keyword")
    parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=2000, help="Max number of posts to fetch (default 2000)")
    parser.add_argument("--subreddit", type=str, default="all", help="Subreddit to search (default 'all')")

    args = parser.parse_args()

    results = fetch_reddit_posts(args.keyword, args.start_date, args.end_date, args.limit, args.subreddit)
    print(f"Fetched {len(results)} posts")
    for r in results[:10]:
        print(r)
