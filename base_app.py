"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from turtle import color, width
import streamlit as st
import streamlit.components.v1 as stc
import joblib, os
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
import time
from PIL import Image
import pickle as pkle
import os.path


# Data dependencies
import pandas as pd
import numpy as np
import re
import base64
from wordcloud import WordCloud, STOPWORDS
from PIL import Image




# Model_map

model_map = {'LinearSVC': 'lsvc_model_pipe.pkl', 'PolynomialSVC': 'psvc_model_pipe.pkl', 'LogisticRegression': 'mlr_model_pipe.pkl', 'MultinomialNB': 'mnb_model_pipe.pkl'}

# Load your raw data
raw = pd.read_csv("train_streamlit.csv", keep_default_na=False)

# creating a sentiment_map and other variables
sentiment_map = {"Anti-Climate": -1, 'Neutral': 0, 'Pro-Climate': 1, 'News-Fact': 2}
type_labels = raw.sentiment.unique()
df = raw.groupby('sentiment')
palette_color = sns.color_palette('dark')

scaler = preprocessing.MinMaxScaler()


def cleaning(tweet):
    """The function uses patterns with regular expression, 'stopwords'
        from natural language processing (nltk) and  tokenize using split method
        to filter and clean each tweet message in a dataset"""

    pattern = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    rem_link = re.sub(pattern, '', tweet)
    rem_punct = re.sub(r'[^a-zA-Z ]', '', rem_link)
    rem_punct = re.sub(r'RT', '', rem_punct)
    word_split = rem_punct.lower().split()
    stops = set(stopwords.words("english"))
    without_stop_sent = ' '.join([t for t in word_split if t not in stops])
    return without_stop_sent



def bag_of_words_count(words, word_dict={}):
    """ this function takes in a list of words and returns a dictionary
        with each word as a key, and the value represents the number of
        times that word appeared"""
    words = words.split()
    for word in words:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    return word_dict


def tags(sentiment_cat=1, iter_hash_num=5, labels=type_labels, dataframe=df, col_type: str = 'hash_tag'):
    sentiment_dict = {}
    counter = 0
    for pp in labels:
        sentiment_dict[pp] = {}
        for row in dataframe.get_group(pp)[col_type]:
            sentiment_dict[pp] = bag_of_words_count(row, sentiment_dict[pp])
    result = {}
    for w in sorted(sentiment_dict[sentiment_cat], key=sentiment_dict[sentiment_cat].get, reverse=True):
        counter += 1
        result[w] = sentiment_dict[sentiment_cat][w]
        if counter >= iter_hash_num:
            break
    return result


def word_grouping(group_word_num=3, sentiment_cat=1, ngram_iter_num=3, dataframe=df):
    ngram_dict = {}
    # converting each word in the dataset into features
    vectorized = CountVectorizer(analyzer="word", ngram_range=(group_word_num, group_word_num),
                                 max_features=1000)  # setting the maximum feature to 8000
    reviews_vect = vectorized.fit_transform(dataframe.get_group(sentiment_cat)['cleaned_tweet'])
    features = reviews_vect.toarray()
    # Knowing the features that are present
    vocab = vectorized.get_feature_names_out()
    # Sum up the counts of each vocabulary word
    dist = np.sum(features, axis=0)

    # For each, print the vocabulary word and the number of times it
    for tag, count in zip(vocab, dist):
        ngram_dict[tag] = count
    # Creating an iteration
    most_pop = iter(sorted(ngram_dict, key=ngram_dict.get, reverse=True))
    result = {}
    for x in range(ngram_iter_num):
        most_pop_iter = next(most_pop)
        result[most_pop_iter] = ngram_dict[most_pop_iter]
        # print(most_pop_iter, ngram_dict[most_pop_iter])
    return result


# """### gif from local file"""
file_happy = open("happy_face.gif", "rb")
contents_happy = file_happy.read()
data_url_happy = base64.b64encode(contents_happy).decode("utf-8")
file_happy.close()

file_sad = open("sad_face.gif", "rb")
contents_sad = file_sad.read()
data_url_sad = base64.b64encode(contents_sad).decode("utf-8")
file_sad.close()

file_news = open("news_face.gif", "rb")
contents_news = file_news.read()
data_url_news = base64.b64encode(contents_news).decode("utf-8")
file_news.close()

file_neutral = open("neutral_face.gif", "rb")
contents_neutral = file_neutral.read()
data_url_neutral = base64.b64encode(contents_neutral).decode("utf-8")
file_neutral.close()


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    # favicon = Image.open("favicon.png")
    st.set_page_config(page_title="Classifier App", page_icon=":hash:", layout="centered")

    # Creates a main title and subheader on your page -
    logo = Image.open("tweet_logo.png")
    st.image(logo)
    #st.title("Eco")
    # st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    menu = ["Home Page", "About Classifier App", "Prediction by Text", "Prediction by File Upload", "Data Exploration", "Company Profile", "About Team"]
    selection = st.sidebar.selectbox("Choose a Page to Display Here 👇", menu)


    if selection == "Home Page":
        st.markdown('')
    elif selection == "About Classifier App":
        st.subheader("Learn How to us the Classifier App")
    elif selection == "Prediction by Text":
        st.subheader("Prediction by Text")
    elif selection == "Data Exploration":
        st.subheader("Exploration of Sentiment and Tweets")
    elif selection == "Prediction by File Upload":
        st.header("Upload File for Prediction")
    elif selection == "Company Profile":
        st.header("Company Profile")
    else:
        st.subheader("About Team")



    #Landing page
    landing = Image.open("backgroundpix.png")
    if selection == "Home Page":
        st.image(landing)#, height=1500)
        # time.sleep(3)
        # st.subheader("Text Classification App") 
        # st.button("Go to next page")

    if selection == "About Classifier App":
        #st.info("This section explains what this app does and how to use it.")
        st.write("""This app was primarily created for tweets expressing belief in climate change. There are seven pages in the app, 
                including the home page, information about classifier apps, predictions by text and files uploaded, data exploration, 
                company profiles, and team information.

                Home Page: The home page is the app's landing page and includes a welcome message and a succinct summary of the app.

                Classifier App: Information The page you are on right now is this one. It includes a detailed explanation of the app as 
                well as usage guidelines.

                Text Prediction: To make a text prediction, enter any text in the textbox underneath the section and press the 
                "Predict" button.""")
        
    #Text Prediction page
    if selection == "Prediction by Text":
        st.info("Type or paste a tweet in the textbox below for the climate change sentiment classification")

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text (Type below)", " ")
        
        model_name = st.selectbox("Choose Model", model_map.keys())
        tweet_process = cleaning(tweet_text)

        st.write('You selected:', model_name)

        if model_name == 'LinearSVC':
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the LinearSVC works goes in here""")

        elif model_name == 'PolynomialSVC':
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the PolynomialSVC model works goes in here""")

        elif model_name == 'LogisticRegression':
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the LogisticRegression model works goes in here""")

        else: 
            with st.expander("See explanation"):
                st.write("""Brief explanation of how the MultinomialNB model works goes in here""")

        
        if st.button("Click to Classify"):
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join(model_map[model_name]), "rb"))
            prediction = predictor.predict([tweet_process])

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            for sen in sentiment_map.keys():
                if sentiment_map.values() == int(prediction):
                    st.success("Text Categorized as: {}".format(sen))

            if prediction == 1:
                st.write(""" **The tweet supports the belief of man-made climate change**""")
                st.markdown(f'<img src="data:image/gif;base64,{data_url_happy}" alt="cat gif">',
                            unsafe_allow_html=True)
            
            elif prediction == -1:
                st.write("""**The tweet do not believe in man-made climate change**""")
                st.markdown(f'<img src="data:image/gif;base64,{data_url_sad}" alt="cat gif">',
                            unsafe_allow_html=True)

            elif prediction == 0:
                st.write("""**The tweet neither supports nor refutes the belief of man-made climate change.**""")
                st.markdown(f'<img src="data:image/gif;base64,{data_url_neutral}" alt="cat gif">',
                            unsafe_allow_html=True)
                
            else:
                st.write("""**The tweet links to factual news about climate change**""")
                st.markdown(f'<img src="data:image/gif;base64,{data_url_news}" alt="cat gif">',
                            unsafe_allow_html=True)
   
    # Building About Team page
    if selection == "About Team":
        st.write("We work with seasoned professionals to give the best product experience")

        st.markdown(" ")
        lista_pic = Image.open("Lista.png")
        nnamdi_pic = Image.open("Nnamdi.jpg")
        othuke_pic = Image.open("Othuke.jpg")


        lista, nnamdi, othuke = st.columns(3)

        lista.success("Founder")
        nnamdi.success("Product Manager")
        othuke.success("Machine Learning Engineer")

        with lista:
            st.header("Lista")
            st.image(lista_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Founder of TechNation.Inc. When it comes to personalizing your online store, nothing is more effective than 
                an About Us page. This is a quick summary of your company's history and purpose, and should provide a clear overview of the 
                company's brand story. A great About Us page can help tell your brand story, establish customer loyalty, and turn your bland 
                ecommerce store into an well-loved brand icon. Most importantly, it will give your customers a reason to shop from your brand.

                In this post, we'll give you three different ways to create a professional about us page for your online store, blog, or other 
                website - use our about us page generator, use the fill-in-the-blank about us template below, or create your own custom page 
                using the about us examples within this article.
                """)

        with nnamdi:
            st.header("Nnamdi")
            st.image(nnamdi_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Nnamdi is a senior product manager with extensive expertise creating high-quality software and a background in user 
                experience design. He has expertise in creating and scaling high-quality products. He has been able to coordinate 
                across functional teams, work through models, visualizations, prototypes, and requirements thanks to his attention to detail.
                
                He frequently collaborates with data scientists, data engineers, creatives, and other professionals with a focus on business. 
                He has acquired expertise in engineering, entrepreneurship, conversion optimization, online marketing, and user experience. 
                He has gained a profound insight of the customer journey and the product lifecycle thanks to that experience.
                """)

        with othuke:
            st.header("Othuke")
            st.image(othuke_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Othuke is a Senor Machine Learning engineer When it comes to personalizing your online store, nothing is more effective than 
                an About Us page. This is a quick summary of your company's history and purpose, and should provide a clear overview of the 
                company's brand story. A great About Us page can help tell your brand story, establish customer loyalty, and turn your bland 
                ecommerce store into an well-loved brand icon. Most importantly, it will give your customers a reason to shop from your brand.

                In this post, we'll give you three different ways to create a professional about us page for your online store, blog, or other 
                website - use our about us page generator, use the fill-in-the-blank about us template below, or create your own custom page 
                using the about us examples within this article.
                """)

        humphrey, valentine, emmanuel = st.columns(3)
        valentine.success("Lead Strategist")
        humphrey.success("Data Scientist")
        emmanuel.success("Success Lead")

        valentine_pics = Image.open("valentine.jpg")
        humphrey_pics = Image.open("humphrey.jpg")
        emmanuel_pics = Image.open("emmanuel.jpg")

        with valentine:
            st.header("Okechukwu")
            st.image(valentine_pics)

            with st.expander("Brief Bio"):
                st.write("""
                Okechukwu When it comes to personalizing your online store, nothing is more effective than 
                an About Us page. This is a quick summary of your company's history and purpose, and should provide a clear overview of the 
                company's brand story. A great About Us page can help tell your brand story, establish customer loyalty, and turn your bland 
                ecommerce store into an well-loved brand icon. Most importantly, it will give your customers a reason to shop from your brand.

                In this post, we'll give you three different ways to create a professional about us page for your online store, blog, or other 
                website - use our about us page generator, use the fill-in-the-blank about us template below, or create your own custom page 
                using the about us examples within this article.
                """)

        with humphrey:
            st.header("Humphrey")
            st.image(humphrey_pics)

            with st.expander("Brief Bio"):
                st.write("""
                Humphery (Osas) Ojo,  an enthusiastic Data Scientist with great euphoria for Exploratory Data Analysis
                (Power-BI, Tableau, Excel, SQL, Python, R) and Machine Learning Engineering(Supervised and Unsupervised Learning), 
                mid-level proficiency in Front-End Web Development(HTML, CSS, MVC, RAZOR, C#).
                """)

        with emmanuel:
            st.header("Emmanuel")
            st.image(emmanuel_pics)

            with st.expander("Brief Bio"):
                st.write("""
                Emmanuel When it comes to personalizing your online store, nothing is more effective than 
                an About Us page. This is a quick summary of your company's history and purpose, and should provide a clear overview of the 
                company's brand story. A great About Us page can help tell your brand story, establish customer loyalty, and turn your bland 
                ecommerce store into an well-loved brand icon. Most importantly, it will give your customers a reason to shop from your brand.

                In this post, we'll give you three different ways to create a professional about us page for your online store, blog, or other 
                website - use our about us page generator, use the fill-in-the-blank about us template below, or create your own custom page 
                using the about us examples within this article.
                """)



    # Building out the prediction page
    if selection == "Prediction by File Upload":
        st.info("Upload a 'one-column' csv file containing tweets of users about their believe on climate change and click on the 'Process' button below to classify the data into the various four various sentiment classes.")

        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if st.button("Process"):
            if data_file is not None:
                df = pd.read_csv(data_file)
                tweet_process = df['message'].apply(cleaning)
                model_name = 'mlr_model_pipe.pkl'
                predictor = joblib.load(open(os.path.join(model_name), "rb"))
                prediction = predictor.predict(tweet_process)
                table = st.table([pd.DataFrame(prediction).value_counts()])
                                
                st.success(table)
                #plt.show()
                #st.pyplot(fig, use_container_width=True) 
                

    if selection == "Data Exploration":
        hash_pick = st.checkbox('Hash-Tag')
        if hash_pick:
            val = st.selectbox("Choose Tag type", ['Hash-Tag', 'Mentions'])
            sentiment_select = st.selectbox("Choose Option", sentiment_map)
            iter_hash_select = st.slider('How many hash-tag', 1, 20, 10)
            if val == 'Hash-Tag':
                st.info("Popular Hast Tags")
            else:
                st.info("Popular Mentions")
            valc = 'hash_tag' if val == 'Hash-Tag' else 'mentions'
            result = tags(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select,
                          col_type=valc)
            source = pd.DataFrame({
                'Frequency': result.values(),
                'Hash-Tag': result.keys()
            })
            val = np.array(list(result.values())).reshape(-1, 1)
            dd = (scaler.fit_transform(val)).reshape(1, -1)
            fig, ax = plt.subplots(1,2, figsize=(10, 3))

            ax[0].bar(data=source, height=result.values(), x= result.keys(), color='#ecc12e')
            ax[0].set_xticklabels(result.keys(), rotation=90)

            mask1 = np.array(Image.open('cloud.png'))
            word_cloud = WordCloud(random_state=1,
                                    background_color='white',
                                    colormap='Set2',
                                    collocations=False,
                                    stopwords = STOPWORDS,
                                    mask=mask1,
                                   width=512,
                                   height=384).generate(' '.join(result.keys()))                       
            ax[1].imshow(word_cloud)
            ax[1].axis("off")
            plt.show()
            st.pyplot(fig, use_container_width=True)


        word_pick = st.checkbox('Word Group(s)')
        if word_pick:
            st.info("Popular Group of Word(s)")
            sentiment_select_word = st.selectbox("Choose sentiment option", sentiment_map)
            word_amt = st.slider('Group of words', 1, 10, 5)
            group_amt = st.slider("Number of Observations", 1, 10, 5)
            word_result = word_grouping(group_word_num=word_amt, ngram_iter_num=group_amt,
                                        sentiment_cat=sentiment_map[sentiment_select_word])
            st.table(pd.DataFrame({
                'Word group': word_result.keys(),
                'Frequency': word_result.values()
            }))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
