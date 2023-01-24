from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import uvicorn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class Item(BaseModel):
    sentence: str
    
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Sign Language API"}


@app.post("/a2sl")
def a2sl(Item: Item):
    text=Item.sentence
    text=text.lower()
    
    words = word_tokenize(text)

    tagged = nltk.pos_tag(words)
    tense = {}
    tense["future"] = len([word for word in tagged if word[1] == "MD"])
    tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
    tense["present_continuous"] = len([word for word in tagged if word[1] in ["VBG"]])

    stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've",'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])

    lr = WordNetLemmatizer()
    filtered_text = []
    for w,p in zip(words,tagged):
        if w not in stop_words:
            if p[1]=='VBG' or p[1]=='VBD' or p[1]=='VBZ' or p[1]=='VBN' or p[1]=='NN':
                filtered_text.append(lr.lemmatize(w,pos='v'))
            elif p[1]=='JJ' or p[1]=='JJR' or p[1]=='JJS'or p[1]=='RBR' or p[1]=='RBS':
                filtered_text.append(lr.lemmatize(w,pos='a'))

            else:
                filtered_text.append(lr.lemmatize(w))

    words = filtered_text
    temp=[]
    for w in words:
        if w=='I':
            temp.append('Me')
        else:
            temp.append(w)
    words = temp
    probable_tense = max(tense,key=tense.get)

    if probable_tense == "past" and tense["past"]>=1:
        temp = ["Before"]
        temp = temp + words
        words = temp
    elif probable_tense == "future" and tense["future"]>=1:
        if "Will" not in words:
                temp = ["Will"]
                temp = temp + words
                words = temp
        else:
            pass
    elif probable_tense == "present":
        if tense["present_continuous"]>=1:
            temp = ["Now"]
            temp = temp + words
            words = temp
    for i in range(len(words)):
        words[i] = words[i].lower()
    
    videos=["0","1","2","3","4","5","6","7","8","9","a",
            "after","again","against","age","all","alone","also","and","ask","at",
            "b","be","beautiful","before","best","better","busy","but","bye",
            "c","can","cannot","change","college","come","computer",
            "d","day","distance","do not","does not",
            "e","eat","engineer","f","fight","finish","from","g","glitter","go",
            "god","gold","good","great","h","hand","hands","happy","hello","help",
            "her","here","his","home","homapage","how","i","invent","it","j","k","keep",
            "l","language","laugh","learn","m","me","more","my","n","name","next",
            "not","now","o","of","on","our","out","p","pretty","q","r","right","s","sad",
            "safe","see","self","sign","so","sound","stay","study","t","talk","television",
            "thank you","thank","that","they","this","those","time","to","type","u","us","v",
            "w","walk","wash","way","we","welcome","what","when","where","which","who","whole",
            "whose","why","will","with","without","words","work","world","wrong","x","y","you",
            "your","yourself","z"]
    filtered_text=[]
    for w in words:
        #splitting the word if its animation is not present in database
        if w not in videos:
            for c in w:
                filtered_text.append(c)
        else:
            filtered_text.append(w)

    for i in range(len(filtered_text)):
        filtered_text[i]=filtered_text[i].title()

    words=filtered_text
    return words
    

@app.post("/post")
def post():

    df=pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQ2bE34VkmJ7cXaFywc5LLKfBYCAmziBuOeRd5DygHnwiQspJ9RG-p05cTQnpizvoOlUYETDwu37NSB/pub?output=csv")
    
    tfidf = TfidfVectorizer(stop_words='english')
    df['description'] = df['description'].fillna("")

    #Construct the required TF-IDF matrix by applying the fit_transform method on the overview 
    overview_matrix = tfidf.fit_transform(df['description'])

    similarity_matrix = linear_kernel(overview_matrix,overview_matrix)

    mapping = pd.Series(df.index,index = df["Title"])

    def recommend_posts_based_on_plot(post_input):
        post_index = mapping[post_input]
        #get similarity values with other posts
        #similarity_score is the list of index and similarity matrix
        similarity_score = list(enumerate(similarity_matrix[post_index]))
        #sort in descending order the similarity score of post inputted with all the other posts
        similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        # Get the scores of the 5 most similar posts. Ignore the first post.
        similarity_score = similarity_score[1:6]
        #return post names using the mapping series
        post_indices = [i[0] for i in similarity_score]
        return (df.iloc[post_indices])
    
    recommeded=recommend_posts_based_on_plot("Major Devender Pal, India’s blade runner")

    #recommeded=df[:][:5]
        
    return {
        "Recommended":recommeded,
        "Post":df
    }
    
if __name__ == "__main__":
    uvicorn.run(app)
