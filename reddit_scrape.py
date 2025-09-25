import praw

reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="forecast_net/0.1 by u/AggravatingChest7621"
)

sub = reddit.subreddit("wallstreetbets")
for post in sub.hot(limit=10):
    print(post.title, post.score, post.url)