from sentiment_analysis import analyze_sentiment
sentence = "DeDora Capital Inc. decreased its holdings in $TSLA by 0.9% during the 2nd quarter. TrueMark Investments LLC grew its position in shares of $KO by 8.8% during the 2nd quarter. Sigma Investment Counselors Inc. boosted its holdings in $ORCL by 4.8% during the second quarter. Arcus Capital Partners LLC lowered its stake in $GLD by 63.3% during the second quarter. Forvis Mazars Wealth Advisors LLC raised its position in shares of $NFLX by 76.7% in the second quarter. Market showing mixed signals as institutional investors adjust their portfolios."

print(analyze_sentiment(sentence)[5])
