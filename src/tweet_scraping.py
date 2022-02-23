# Python Script to Extract tweets of a
# particular Hashtag using Tweepy and Pandas

# import modules
import pandas as pd
import tweepy

# function to perform data extraction
def scrape(words, date_since, numtweet):
    # Enter your own credentials obtained
    # from your developer account
    consumer_key = "81vbRbjOgE5IfMlpc6Th3hAfF"
    consumer_secret = "dz4bKqd2qHTbC4u9WyflgOEtQA4ypl3Wf0FRIhMUoJePJbO5nM"
    access_key = "1473799519011753989-KwZASLFzHJaedhxmicKNaXkOAoyPAD"
    access_secret = "Hi94rN2sLJ4HJDEqcNahtuNP1PLePcEFcui3ks7kFipyN"


    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # Creating DataFrame using pandas
    db = pd.DataFrame(columns=['username',
                                'description',
                                'location',
                                'following',
                                'followers',
                                'totaltweets',
                                'retweetcount',
                                'text',
                                'hashtags'])

    # We are using .Cursor() to search
    # through twitter for the required tweets.
    # The number of tweets can be
    # restricted using .items(number of tweets)
    tweets = tweepy.Cursor(api.search_tweets,
                            words, lang="en",
                            since_id=date_since,
                            tweet_mode='extended').items(numtweet)

    # .Cursor() returns an iterable object. Each item in
    # the iterator has various attributes
    # that you can access to
    # get information about each tweet
    list_tweets = [tweet for tweet in tweets]

    # Counter to maintain Tweet Count
    i = 1

    # we will iterate over each tweet in the
    # list for extracting information about each tweet
    for tweet in list_tweets:
        username = tweet.user.screen_name
        description = tweet.user.description
        location = tweet.user.location
        following = tweet.user.friends_count
        followers = tweet.user.followers_count
        totaltweets = tweet.user.statuses_count
        retweetcount = tweet.retweet_count
        hashtags = tweet.entities['hashtags']

        # Retweets can be distinguished by
        # a retweeted_status attribute,
        # in case it is an invalid reference,
        # except block will be executed
        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text
            hashtext = list()
            for j in range(0, len(hashtags)):
                hashtext.append(hashtags[j]['text'])

                # Here we are appending all the
                # extracted information in the DataFrame
            ith_tweet = [username, description,
                        location, following,
                        followers, totaltweets,
                        retweetcount, text, hashtext]
            db.loc[len(db)] = ith_tweet

    filename = 'scraped_tweets.csv'

    # we will save our database as a CSV file.
    db.to_csv(filename)    