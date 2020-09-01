import tweepy
import pandas as pd 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

# Criando as chaves de autenticação
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

df =[]
#https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/overview/tweet-object
for tweet in tweepy.Cursor(api.search, q="*", lang='pt-br').items(1000):
    if 'RT @' not in tweet.text: #ignorando Retweets
        df.append([tweet.user.id, tweet.text, tweet.source, tweet.created_at, tweet.retweet_count, 
                    tweet.favorite_count, tweet.user.followers_count, tweet.user.friends_count, tweet.user.verified])

#Criar um DataFrame com os dados obtidos do Twitter
dataframe = pd.DataFrame(df, columns=['id_user','texto', 'plataforma', 'data', 
                            'retweet','favorito', 'seguidores', 'amigos', 'verificado'])
dataframe.to_csv('TwitterBrasil.csv')#salvando o DataFrame em um arquivo CSV

#Contar quais palavras são mais utilizadas em tweets em Pt-br
cv = CountVectorizer()
count_matrix = cv.fit_transform(df.texto)

word_count = pd.DataFrame(cv.get_feature_names(), columns=["texto"])
word_count["count"] = count_matrix.sum(axis=0).tolist()[0]
word_count = word_count.sort_values("count", ascending=False).reset_index(drop=True)
print(word_count[:60])