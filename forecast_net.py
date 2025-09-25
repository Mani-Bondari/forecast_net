"""_summary_
forecast_net is designed to use a variety of information sources to forcast trends in the
stock market over a 7 (trading) day period. while hedgefunds billion dollar firms have access
to financial information at lower latency than the average folk, us average folk can take
advantage of the vast variety of media and information channels at our disposal such as:
- yahoo finance
- stocktwits
- reddit
- instagram comments
- threads
- tik tok comments
- etc

with a selective picky scraping system using a mix of api's, web-scraping, custom ranking
algorithms etc. we can collect a vast variety of information from a wide array of sources
broadening our model in the breadth aspect as opposed to the depth that large firms have.
this heightened breadth should make up for the increased latency of the acquisition of this 
information.


financial info we will be using:
--------------------------------
60 day return history of the stock (close value - open value) represented as a tensor (B, 30, 1)

30 day volume history (volume for each of the previous 30 days ligned up with the above) (B, 30, 1)

30 day market value history (related to the return history, 
but can let us know if there were any stock splits which can impact the forecast) (B, 30, 1)

yahoo finance's calculation of the most recent beta value to express the volatility of the stock (B, 1, 1)

30 day history of the P/E value used to calculate how the stock moves in general (B, 30, 1)

social info we will be using:
-------------------------------
reddit:
we will scrape a list of 5000 top reddit posts from a certain date interval related to the ticker 
symbol in question, we will use the following ranking algorithm to rank each of the posts to filter out noise:

u = upvotes
c = # comments
l = decay rate of decay function e^-l(dt)
k = author karma
r = sentiment relevence score

Score(post) = (a * norm(u) + b * norm(c)) * (1 + l * log(1 + k)) * e^(-l(dt)) * r

a, b, l, r are all learnable parameters by the model, so it can learn over a long period of time and looking 
at data what factors contribute to a higher reliability and relevance score for the task at hand.



    
"""