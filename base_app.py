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
from streamlit_option_menu import option_menu
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

model_map = {'LinearSVC': 'resources/lsvc_model_pipe.pkl', 'PolynomialSVC': 'resources/psvc_model_pipe.pkl', 
'LogisticRegression': 'resources/mlr_model_pipe.pkl', 'MultinomialNB': 'resources/mnb_model_pipe.pkl'}

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
file_happy = open("resources/imgs/happy_face.gif", "rb")
contents_happy = file_happy.read()
data_url_happy = base64.b64encode(contents_happy).decode("utf-8")
file_happy.close()

file_sad = open("resources/imgs/sad_face.gif", "rb")
contents_sad = file_sad.read()
data_url_sad = base64.b64encode(contents_sad).decode("utf-8")
file_sad.close()

file_news = open("resources/imgs/news_face.gif", "rb")
contents_news = file_news.read()
data_url_news = base64.b64encode(contents_news).decode("utf-8")
file_news.close()

file_neutral = open("resources/imgs/neutral_face.gif", "rb")
contents_neutral = file_neutral.read()
data_url_neutral = base64.b64encode(contents_neutral).decode("utf-8")
file_neutral.close()


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    st.set_page_config(page_title="Classifier App", page_icon=":hash:", )

    # Creates a main title and subheader on your page -
    logo = Image.open("resources/imgs/tweet_logo.png")
    st.image(logo)
    #st.title("Eco")
    # st.subheader("Climate change tweet classification")

    # Design horizontal bar
    menu = ["Home", "EDA", "Prediction", "About"]
    selection = option_menu( menu_title=None,
    options=menu,
    icons=["house", "graph-up", "textarea-t",  "file-person"],
    orientation='horizontal',
    styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "orange", "font-size": "25px",  },
                "nav-link": {
                    "font-size": "20px",
                    "text-align": "center",
                    "margin": "5px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )    



    if selection == "Home":
        st.markdown('')

    elif selection == "Prediction":
        st.subheader("Prediction")

    elif selection == "EDA":
        st.subheader("Exploration of Sentiment and Tweets")

    else:
        st.subheader('')



    #Landing page
    landing = Image.open("resources/imgs/backgroundpix.png")
    if selection == "Home":
        st.image(landing)#, height=1500

    
    #Text Prediction page
    if selection == "Prediction":
        menu = ['Single Text Prediction', 'Batch Prediction']
        type = st.sidebar.selectbox("Choose Prediction Type 👇", menu)

        if type == 'Single Text Prediction':

            st.info("Type or paste a tweet in the textbox below for the climate change sentiment classification")

            # Creating a text box for user input
            tweet_text = st.text_area("Enter Text (Type below)", " ")
        
            model_name = st.selectbox("Choose Model", model_map.keys())
            tweet_process = cleaning(tweet_text)

            st.write('You selected:', model_name)

            if model_name == 'LinearSVC':
                with st.expander("See explanation"):
                    st.write("""Linear Support Vector Classification.

Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.""")

            elif model_name == 'PolynomialSVC':
                with st.expander("See explanation"):
                    st.write("""Polynomial Kernel

In machine learning, the polynomial kernel is a kernel function commonly used with support vector machines (SVMs) and other kernelized models, that represents the similarity of vectors (training samples) in a feature space over polynomials of the original variables, allowing learning of non-linear models.

Intuitively, the polynomial kernel looks not only at the given features of input samples to determine their similarity, but also combinations of these. In the context of regression analysis, such combinations are known as interaction features. The (implicit) feature space of a polynomial kernel is equivalent to that of polynomial regression, but without the combinatorial blowup in the number of parameters to be learned. When the input features are binary-valued (booleans), then the features correspond to logical conjunctions of input features""")

            elif model_name == 'LogisticRegression':
                with st.expander("See explanation"):
                    st.write("""Logistic Regression (aka logit, MaxEnt) classifier.

In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’. (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ solvers.)

This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that regularization is applied by default. It can handle both dense and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance; any other input format will be converted (and copied).

The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization with primal formulation, or no regularization. The ‘liblinear’ solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty. The Elastic-Net regularization is only supported by the ‘saga’ solver.""")

            else: 
                with st.expander("See explanation"):
                    st.write("""Naive Bayes classifier for multinomial models.

The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.""")

        
            if st.button("Predict"):
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
                    st.info(""" **This tweet is categorized as Pro: The tweet supports the belief of man-made climate change**""")
                    # st.markdown(f'<img src="data:image/gif;base64,{data_url_happy}" alt="cat gif">',
                    #         unsafe_allow_html=True)
            
                elif prediction == -1:
                    st.info("""**This tweet is categorized as Anti: The tweet do not believe in man-made climate change**""")
                    # st.markdown(f'<img src="data:image/gif;base64,{data_url_sad}" alt="cat gif">',
                    #         unsafe_allow_html=True)

                elif prediction == 0:
                    st.info("""**This tweet is categorized as Neutral: The tweet neither supports nor refutes the belief of man-made climate change.**""")
                    # st.markdown(f'<img src="data:image/gif;base64,{data_url_neutral}" alt="cat gif">',
                    #         unsafe_allow_html=True)
                
                else:
                    st.info("""**This tweet is categorized as News: The tweet links to factual news about climate change**""")
                    # st.markdown(f'<img src="data:image/gif;base64,{data_url_news}" alt="cat gif">',
                    #         unsafe_allow_html=True)
        else:
            st.info("Upload a csv file containing 2 columns; 'tweetid', 'message'. Click on the 'Predict' button below to classify the data into the various four various sentiment classes.")

            data_file = st.file_uploader("Upload CSV",type=['csv'])
            if st.button("Predict"):
                if data_file is not None:
                    df = pd.read_csv(data_file)
                    tweet_process = df['message'].apply(cleaning)
                    model_name = 'resources/mlr_model_pipe.pkl'
                    predictor = joblib.load(open(os.path.join(model_name), "rb"))
                    prediction = predictor.predict(tweet_process)
                    table_2 = pd.DataFrame(prediction).value_counts()
                    sentiment_dict = {2: 'News', 1: 'Pro', 0: 'Neutral', -1:'Anti'}
                    Final_Table = {'tweetid': df.tweetid, 'sentiment': pd.Series(np.round(prediction, 0)).map(sentiment_dict)}
                    pred_df = pd.DataFrame(data=Final_Table)
                    dict_prediction = {'Sentiments': ['News', 'Pro-climate', 'Neutral', 'Anti-climate'], 
                                        'Predictions': [table_2[2], table_2[1], table_2[0], table_2[-1]],
                                        'Percentages of sentiments': [table_2[2]/table_2.sum(), table_2[1]/table_2.sum(), table_2[0]/table_2.sum(), table_2[-1]/table_2.sum()]}               

                    # table = st.table(pred_df['sentiment'].map(sentiment_dict).value_counts())
                    st.write("""
                            **The data uploaded contains {} number of tweets and classified into the following sentiments:**
                            - {} tweets are linked to factual news about climate change;
                            - {} tweets supports the belief of man-made climate change;
                            - {} tweets neither supports nor refutes the belief of man-made climate change;
                            - {} tweets does not believe in man-made climate change
                            """.format(df.shape[0], table_2[2], table_2[1], table_2[0], table_2[-1]))

                    st.info("See a summary table of the sentiment classification below 👇:")
                    pd_pred = pd.DataFrame(dict_prediction)
                    pd_pred  

                    st.info("See a bar chart showing the tweets sentiment classification below 👇:")
                    pic = st.bar_chart(pred_df['sentiment'].value_counts())

                    with st.expander("See full table", ):
                        pred_df

                    @st.cache
                    def convert_df(df):
                        return df.to_csv().encode('utf-8')


                    pred_csv = convert_df(pred_df)

                    st.download_button(
                        "Press to Download csv file",
                        pred_csv,
                        "file.csv",
                        "text/csv",
                        key='download-csv'
                        )

                    # st.success(table)

   
    # Building About Team page
    if selection == "About":
        menu = ['Documentation', 'About Team']
        type = st.sidebar.selectbox("Choose what you want to learn about 👇", menu, )

        if type == 'Documentation':
            st.subheader("**App Documentation**: Learn How to use the Classifier App")
            # time.sleep(3)
            # st.subheader("Text Classification App") 
            # st.button("Go to next page")
            st.write("""
                    This app was primarily created for tweets expressing belief in climate change. There are four pages in the app which 
                                includes; `home page`, `predictions`, `Exploratory Data Analysis` and `About`.

                    - **`Home`**: The home page is the app's landing page and includes a welcome message and a succinct summary of the app.
                    
                    - **`EDA`**: The EDA section, which stands for Explanatory Data Analysis, gives you the chance to explore your data. 
                                Based on the number of hash-tags and mentions in the tweet that have been gathered, it also displays graphs of various groups of 
                                words in the dataset, giving you a better understanding of the data you are working with.

                    - **`Prediction`**: This page is where you use the main functionality of the app. It contains two subpages which are: 
                                `Single Text Prediction` and `Batch Prediction`
                    
                    - **`Single Text Prediction`**: You can predict the sentiment of a single tweet by typing or pasting it on the text prediction 
                                page. Enter any text in the textbox beneath the section, then click "Predict" to make a single tweet prediction.
                    
                    - **`Batch Prediction`**: You can make sentiment predictions for batches of tweets using this section. It can process multiple tweets in a batch from a `.csv` 
                                file with at least two columns named `message` and `tweetid` and categorize them into different tweet sentiment groups. To predict by file up, 
                                click on the `browse file` button to upload your file, then click on process to do prediction. A thorough output of the prediction will be provided, 
                                including a summary table and the number of tweets that were categorised under each sentiment class.
                    
                    See sample of the csv file to be uploaded below 👇: 
                    """)
            csv = Image.open("resources/imgs/csv_sample.png")
            st.image(csv)

            st.write("""
                    - **`About`**: The About page also has two sub-pages;  `Documentation` and `About Team` page.
                   
                    - **`Documentation`**: This is the current page. It includes a detailed explanation of the app as well as usage guidelines on
                            how to use this app with ease.
                    
                    - **`About Team`**: This page gives you a brief summary of the experience of the team who built and manages the app.
                    """)
        
        else:
            st.subheader("About Team")
            st.markdown(" ")
            lista_pic = Image.open("resources/imgs/Lista.png")
            nnamdi_pic = Image.open("resources/imgs/Nnamdi.jpg")
            othuke_pic = Image.open("resources/imgs/Othuke.jpg")
            valentine_pics = Image.open("resources/imgs/valentine.jpg")
            humphrey_pics = Image.open("resources/imgs/humphrey.jpg")
            emmanuel_pics = Image.open("resources/imgs/emmanuel.jpg")


            st.header("Lista - Founder")
            lista, text1 = st.columns((1,2))
        
            with lista:
                st.image(lista_pic)

            with text1:
                st.write("""
                    Lista Abutto is the Founder of Nonnel Data Solution Ltd. she is currently a senior website developer with a background in 
                    soft development, information systems security, digital marketing, and data science.  She has majorly worked in the medical, 
                    educational, government, and hospitality niches with both established and start-up companies. 

                    She is currently pursuing a master's in business administration. 
                    """)

            st.header("Nnamdi - Product Manager")
            nnamdi, text2 = st.columns((1,2))
            
            with nnamdi:
                st.image(nnamdi_pic)

            with text2:
                st.write("""
                    Nnamdi is a senior product manager with extensive expertise creating high-quality software and a background in user 
                    experience design. He has expertise in creating and scaling high-quality products. He has been able to coordinate 
                    across functional teams, work through models, visualizations, prototypes, and requirements thanks to his attention to detail.
                    
                    He frequently collaborates with data scientists, data engineers, creatives, and other professionals with a focus on business. 
                    He has acquired expertise in engineering, entrepreneurship, conversion optimization, online marketing, and user experience. 
                    He has gained a profound insight of the customer journey and the product lifecycle thanks to that experience.
                    """)
            
            st.header("Othuke - Chief Machine Learning Engineer")
            othuke, text3 = st.columns((1, 2))
            
            with othuke:
                st.image(othuke_pic)

            with text3:
                st.write("""
                Othuke is a full-stack Data professional offering descriptive and prescriptive analytics. Worked in pivotal roles that required 
                simplifying the business functions, revenue stream, sales forecasting, competitive analysis, and risk management of Business. 
                
                An AI/Data Scientist with a background in Engineering who posses a sound knowledge of Drone development, Chatbot development, 
                Database Management, Python Programming, Data Analytics, Machine Learning, Artificial Intelligence and Cloud Computing. 
                I'm very passionate about building AI, IT and Engineering solutions to help solve existing business and societal needs. 
                    """)

            st.header("Okechukwu - Lead Strategist")
            valentine, text4 = st.columns((1,2))
                    

            with valentine:
                st.image(valentine_pics)

                with text4:
                    st.write("""
                    Njoku Okechukwu Valentine, a passionate problem solver armed with critical thinking with proficiency in Excel, Powerbi, 
                    SQL and Data science and Machine Learning using Python based technologies. Mid-level Flask Developer, and automation engineer.
                    """)

            st.header("Humphrey - Data Scientist")
            humphrey, text5 = st.columns((1,2))
            
            
            with humphrey:
                st.image(humphrey_pics)

            with text5:
                st.write("""
                    Humphery (Osas) Ojo,  an enthusiastic Data Scientist with great euphoria for Exploratory Data Analysis
                    (Power-BI, Tableau, Excel, SQL, Python, R) and Machine Learning Engineering(Supervised and Unsupervised Learning), 
                    mid-level proficiency in Front-End Web Development(HTML, CSS, MVC, RAZOR, C#).
                    """)

            st.header("Emmanuel - Customer Success")
            emmanuel, text6 = st.columns((1,2))
            
            with emmanuel:
                st.image(emmanuel_pics)

            with text6:
                st.write("""
                    Emmanuel When it comes to personalizing your online store, nothing is more effective than 
                    an About Us page. This is a quick summary of your company's history and purpose, and should provide a clear overview of the 
                    company's brand story. A great About Us page can help tell your brand story, establish customer loyalty, and turn your bland 
                    ecommerce store into an well-loved brand icon. Most importantly, it will give your customers a reason to shop from your brand.
                    """)


                

    if selection == "EDA":
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

            mask1 = np.array(Image.open('resources/imgs/cloud.png'))
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
