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


def get_covidcast_data():
    mask_since_date = datetime.strptime("2020-10-09", '%Y-%m-%d')
    vaccine_since_date = datetime.strptime("2020-12-27", '%Y-%m-%d')
    mask_until_date = datetime.strptime("2021-2-12", '%Y-%m-%d')
    vaccine_until_date = datetime.strptime("2021-2-25", '%Y-%m-%d')
    
    mask_state = covidcast.signal("fb-survey", "smoothed_wearing_mask", mask_since_date, mask_until_date, "state")  
    vaccine_state = covidcast.signal("fb-survey", "smoothed_accept_covid_vaccine", vaccine_since_date, vaccine_until_date, "state")
    
    get_national_data(mask_state, "mask")
    get_national_data(vaccine_state, "vaccine")

def get_national_data(data, model):
    state = data.copy()
    state = state[["geo_value", "time_value", "value", "sample_size"]]
    positive = state["sample_size"] * state["value"] / 100
    negative = state["sample_size"] * (100 - state["value"]) / 100
    state["positive"] = positive
    state["negative"] = negative
    nation = state.groupby("time_value").sum()
    nation["value"] = nation["positive"] / (nation["positive"] + nation["negative"]) * 100
    nation["geo_value"] = "us"
    nation.reset_index(inplace=True)
    
    data = state.append(nation, ignore_index=True)
    data.sort_values(["time_value", "geo_value"], inplace=True)
    data = data.drop(columns=["positive","negative"], axis=1).round(4)
    data.reset_index(drop=True, inplace=True)
    data.to_csv("covidcast_{}.csv".format(model))
    get_trends_data("covidcast_{}.csv".format(model), model)
    
def get_trends_data(data_file, trend):
    covidcast_data = pd.read_csv(data_file)
    trends_data = pd.read_csv("trends_{}/{}_us.csv".format(trend, trend), header=1)
    trends_data.rename(columns={trends_data.columns[0]: "time_value", trends_data.columns[1]: "trend"}, inplace=True)
    trends_data["geo_value"] = "us"

    import os
    file_list = os.listdir("trends_{}".format(trend))
    for file in file_list:
        if file != "{}_us.csv".format(trend):
            state = file.split(".")[0][-2:]
            state_data = pd.read_csv("trends_{}/{}".format(trend, file), header=1)
            state_data.rename(columns={state_data.columns[0]: "time_value", state_data.columns[1]: "trend"}, inplace=True)
            state_data["geo_value"] = state
            trends_data = trends_data.append(state_data, ignore_index=True)

    trends_data.sort_values(["time_value", "geo_value"], inplace=True)
    trends_data.reset_index(drop=True, inplace=True)
    
    mask_all = covidcast_data.merge(trends_data, how="inner", on=["time_value", "geo_value"]).reset_index(drop=True)
    mask_all.drop(mask_all.columns[0], axis=1, inplace=True)
    mask_all.to_csv("{}_all.csv".format(trend))


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
    
def analyze(user, dataset, from_date, target_date, location, use_trends):
    location = location.lower()
    
    tweets = []
    if dataset.lower() == 'vaccine':
        tweets = get_tweets(user=user, keyword='(covid-19 vaccine OR #covid19 vaccine)', since=from_date, lang='lang:en')
        if len(tweets) <= 0:
            print("{} has not made a Covid-19 vaccine related Tweet since {}.".format(user, from_date))
            quit()
    else:
        tweets = get_tweets(user=user, keyword='(mask OR masks)', since=from_date, lang='lang:en')
        if len(tweets) <= 0:
            print("{} has not made a mask related Tweet since {}.".format(user, from_date))
            quit()
    
    X, y, likeCount, retweetCount, replyCount, quoteCount, enc, sample_size = process_data(tweets, use_trends, location)
    train_X, train_y, test_X, test_y = split_data(X, y)
    model = train_model(train_X, train_y, test_X, test_y)
    baseline, tweeted = predict(model, target_date, location, use_trends, likeCount, retweetCount, replyCount, quoteCount, enc)
    
    return baseline, tweeted, sample_size
#     location = location.upper()
#     if dataset.lower() == 'vaccine':
#         print("Predicted baseline vaccine acceptance likelihood in {}: {}".format([location], baseline))
#         print("Predicted vaccine acceptance likelihood in {} if a Tweet was made by {}: {}".format([location], [user], tweeted))
#     else:
#         print("Predicted baseline mask wearing likelihood in {}: {}".format([location], baseline))
#         print("Predicted mask wearing likelihood in {} if a Tweet was made by {}: {}".format([location], [user], tweeted))

#     baseline_increase = baseline * sample_size / 100
#     tweeted_increase = tweeted * sample_size / 100
#     change = tweeted_increase - baseline_increase
#     increase = "[more]"
#     if change < 0:
#         increase = "[less]"
        
#     if dataset.lower() == 'vaccine':
#         print("With a average sample size of {}, Tweeting may cause {} {} people to accept vaccines.".format([round(sample_size)], [abs(int(change))], increase))
#     else:
#         print("With a average sample size of {}, Tweeting may cause {} {} people to wear masks.".format([round(sample_size)], [abs(int(change))], increase))
