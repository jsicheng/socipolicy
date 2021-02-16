import snscrape.modules.twitter as sntwitter
from datetime import date, datetime
import covidcast
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit

def get_tweets(user, keyword, since, lang='lang:en'):
    user = 'from:' + user + ' '
    keyword += ' '
    since = 'since:' + since + ' '
    until = 'until:2021-01-31 '
    maxTweets = 1000
    tweets = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(user + keyword + since + until + lang).get_items()) :
            if i > maxTweets :
                break
            tweets.append({
#                 "username":tweet.user.username,
#                 "url":tweet.url,
                "time_value":tweet.date.strftime("%Y-%m-%d"),
                "likeCount":tweet.likeCount,
                "retweetCount":tweet.retweetCount,
                "replyCount":tweet.replyCount,
                "quoteCount":tweet.quoteCount
            })
    return tweets

def get_data(location, since, until):
    since_date = datetime.strptime(since, '%Y-%m-%d')
    until_date = datetime.strptime(until, '%Y-%m-%d')
    
    if location == "us":
        #data = covidcast.signal("indicator-combination", "confirmed_incidence_prop", since_date, until_date, location)
        data = covidcast.signal("fb-survey", "smoothed_wearing_mask", since_date, until_date, location)
    else:
        #data = covidcast.signal("indicator-combination", "confirmed_incidence_prop", since_date, until_date, "state").query('geo_value == "{}"'.format(location))
        data = covidcast.signal("fb-survey", "smoothed_wearing_mask", since_date, until_date, "state").query('geo_value == "{}"'.format(location))

    return data

def process_data(tweets, use_trends, geo_value, mask_file="mask_all.csv"):
    mask_data = pd.read_csv(mask_file)
    sample_size = mask_data.query('geo_value == "{}"'.format(geo_value)).groupby("geo_value").mean()["sample_size"].sum()
    training_data = mask_data[["time_value", "geo_value", "trend"]].reset_index(drop=True)
    if use_trends == -1:
        training_data = mask_data[["time_value", "geo_value"]].reset_index(drop=True)
    tweet_data = pd.DataFrame.from_records(tweets).groupby("time_value").sum()
    likeCount = tweet_data["likeCount"].mean()
    retweetCount = tweet_data["retweetCount"].mean()
    replyCount = tweet_data["replyCount"].mean()
    quoteCount = tweet_data["quoteCount"].mean()
    
    training_data = training_data.merge(tweet_data, how="left", on="time_value").reset_index(drop=True).fillna(0)
    training_data["time_value"] = training_data["time_value"].apply(lambda x: int(x.replace("-", "")))

    enc = OneHotEncoder()
    enc_df = pd.DataFrame(enc.fit_transform(training_data[["geo_value"]]).toarray())
    training_data = training_data.join(enc_df).drop(training_data[["geo_value"]], axis=1)

    X = np.array(training_data)
    y = np.array(mask_data["value"])
    
    return X, y, likeCount, retweetCount, replyCount, quoteCount, enc, sample_size

def split_data(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
    return train_X, train_y, test_X, test_y

def train_model(train_X, train_y, test_X, test_y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators = 50, random_state=1)
    model.fit(train_X, train_y);

    predictions = model.predict(test_X)
    errors = abs(predictions - test_y)
    print('Mean Absolute Error:', round(np.mean(errors), 4))
    return model
    
def predict(model, time_value, geo_value, trend, likeCount, retweetCount, replyCount, quoteCount, enc):
    baseline = pd.DataFrame({"time_value":[time_value.replace("-", "")], "geo_value":[geo_value], "trend":[trend], "likeCount":0, "retweetCount":0, "replyCount":0, "quoteCount":0})
    if trend == -1:
        baseline = pd.DataFrame({"time_value":[time_value.replace("-", "")], "geo_value":[geo_value], "likeCount":0, "retweetCount":0, "replyCount":0, "quoteCount":0})
    enc_df = pd.DataFrame(enc.transform(baseline[["geo_value"]]).toarray())
    baseline = baseline.join(enc_df).drop(baseline[["geo_value"]], axis=1)
    baseline = model.predict(baseline).round(4)
    
    
    query = pd.DataFrame({"time_value":[time_value.replace("-", "")], "geo_value":[geo_value], "trend":[trend], "likeCount":likeCount, "retweetCount":retweetCount, "replyCount":replyCount, "quoteCount":quoteCount})
    if trend == -1:
        query = pd.DataFrame({"time_value":[time_value.replace("-", "")], "geo_value":[geo_value], "likeCount":likeCount, "retweetCount":retweetCount, "replyCount":replyCount, "quoteCount":quoteCount})
    enc_df = pd.DataFrame(enc.transform(query[["geo_value"]]).toarray())
    query = query.join(enc_df).drop(query[["geo_value"]], axis=1)
    tweeted = model.predict(query).round(4)
    
    return baseline, tweeted
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='sociolicy.py', description='Computes the influence value of a twitter user on covid-19 policy compliance.')
    
    parser.add_argument('-u', '--user', action='store', default='@CDCgov', type=str, help='twitter user to be analyzed')
    parser.add_argument('-f', '--from_date', action='store', default='2020-10-09', type=str, help='beginning date to extract tweets in Y-m-d format')
    parser.add_argument('-t', '--target_date', action='store', default='2021-02-1', type=str, help='date to predict mask wearing liklihood in Y-m-d format')
    parser.add_argument('-l', '--location', action='store', default='us', type=str, help='use us or state (\'ca\', \'ny\', etc) covid-19 data')
    parser.add_argument('-gt', '--use_trends', action='store', default=-1, type=int, help='use google trend value, use -1 to disable')

    args = parser.parse_args()
    
    tweets = get_tweets(user=args.user, keyword='(mask OR masks)', since=args.from_date, lang='lang:en')
    if len(tweets) <= 0:
        print("{} has not made a mask related Tweet since {}.".format(args.user, args.from_date))
        quit()
    
    X, y, likeCount, retweetCount, replyCount, quoteCount, enc, sample_size = process_data(tweets, args.use_trends, args.location)
    train_X, train_y, test_X, test_y = split_data(X, y)
    model = train_model(train_X, train_y, test_X, test_y)
    baseline, tweeted = predict(model, args.target_date, args.location, args.use_trends, likeCount, retweetCount, replyCount, quoteCount, enc)
    
    print("Predicted baseline mask wearing likelihood in {}: {}".format([args.location], baseline))
    print("Predicted mask wearing likelihood in {} if a Tweet was made by {}: {}".format([args.location], [args.user], tweeted))
    
    baseline_increase = baseline * sample_size
    tweeted_increase = tweeted * sample_size
    change = tweeted_increase - baseline_increase
    increase = "[more]"
    if change < 0:
        increase = "[less]"
    print("With a average sample size of {}, Tweeting will cause {} {} people to wear masks.".format([round(sample_size)], [abs(int(change))], increase))