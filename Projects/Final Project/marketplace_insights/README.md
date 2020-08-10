# Market Place Insights

## Food for Thought & Thought for Food

## Erin Werner, Dhileeban Kumaresan, Jocelyn Lu | W207 Summer 2020 | Thursday, 6:30PM 

### Motivation 

Our goal is to improve the Amazon Fine Foods marketplace experience by helping both consumers & retailers glean better insights on the items they buy or sell. 

For some background, the public marketplace today has evolved into an environment where a large seller-base meets a large potential buyer base. It is a many-to-many model. However this is a fragmented system as not all buyers will interact with all sellers. For small businesses to succeed as vendors, and for customers to get value out of the marketplace, we need to understand the best way to connect everything.

We intend to leverage machine learning algorithms in order to provide services that would benefit both ends of the marketplace. These improvements can help lead retailers to more sales and, thus, increased profits while simultaneously satisfying the more nuanced needs of their customer base. Our goals is to take a holistic approach in how we build our machine learning models based on our data. In order to accomplish this, we will consider sentiment analysis and the helpfulness score of the review in addition to the raw star rating in order to make recommendations.


### Our Contribution

Our contribution will largely focus on providing services oriented around improving the experience of both buyers and sellers. 

For *buyers,* we want to curate best fit items, which would recommend products related to what you’ve bought. We also want to utilize purchasing habits and recommend products that similar users to you also bought and liked. This can be accomplished by building a recommender system.

In regard to the *sellers*, we want to interpret the product reviews, through both the helpfulness score and the reviews of the customer. This will help sellers to improve their products or produce new ones in order to better cater to their customers. This can be accomplished by doing topic modeling and sentiment analysis of reviews.

#### Data Processing and Feature Engineering

* [**EDA, Sampling Exploration, and Text Cleanup**](eda)
* [**Custom Word Embedding Model**](feature_engineering/review2vec.ipynb)

We will go through our standard EDA in this module to understand our data and how we may approach the problems we want to solve. In this process, we will also tackle a few data issues and do some feature engineering to provide our models with better data. 

The first step is to handle the text cleanup and preprocessing for the corpora of reviews. As the data are user inputted text reviews that are scraped from the internet, we end up with a plethora of dirty text issues including extra HTML tags and poorly encoded or unescaped characters. We will also do some standard text processing like removing stopwords and lemmanizing the text to reduce our vocabulary size while gleaning as much information from the text as we can.

The next challenge we have is related to the imbalance of positive vs. negative sentiment reviews. In our dataset, the former vastly outweighs the latter. We will explore some different sampling and class balancing techniques, and show that oversampling by synthetically generating data both balances our classes as well as provides our models with clearer decision boundaries. 

Lastly, one method of featurizing/vectorizing our text features is by using word embeddings. We can explore using pre-trained out-of-the-box word embeddings in the [Sentiment Analysis](#Sentiment-Analysis) section. However, we also take it a step further and train a new word embedding model directly on our review corpora. This gives the benefit of maintaining word similarity that is specific to our domain (in this case, food reviews), and limits the vocabulary that the word embeddings can create. We will use both methods when comparing models for sentiment analysis. 

#### Topic Modeling

* [**Topic Modeling**](topic_modeling/topic_modeling.ipynb)

Topic modeling is an unsupervised machine learning technique that's capable of scanning a set of documents, detecting word and phrase patterns within them, and automatically clustering word groups and similar expressions that best characterize a set of documents. A "topic" consists of a cluster of words that frequently occur together. Using contextual clues, topic models can connect words with similar meanings and distinguish between uses of words with multiple meanings.

We will explore different methods of topic modeling using a latent Dirichlet distribution, and evaluate the goodness of fit of the topics for the corpora of reviews. We will also show how sellers may be able to use the results of the topics generated to determine gaps in their products and find better consumer-product fit. 

#### Sentiment Analysis

* **Sentiment Analysis**
  * [Ensemble](sentiment/ReviewExtraction.ipynb)
  * [GloVE + Deep Learning](sentiment/SentimentAnalysisWordEmbedding.ipynb)
  * [Word2Vec + Deep Learning](sentiment/sentiment_trained_review2vec_lstm_custom.ipynb)

Sentiment analysis is the interpretation and classification of emotions (positive or negative) within text data using text analysis techniques. We have explored several machine learning algorithms to predict the sentiment as well as the helpfulness of the reviews in the data. We have implemented four different vectorization methods to convert text data to input into our machine learning algorithms, such as Count and tf-idf vectorization. Additionally, for the neural network models, we used word embedding.

#### Recommender Systems

* [**Recommender Systems**](rec_sys/W207%20-%20Final%20Project%20-%20Recommender%20Systems.ipynb)
  
This file covers the development of the overall Recommender System process. This includes Network Analysis, Collaborative Filtering with the Pearson Coefficient and Neighborhood Grouping, Latent Factorization, as well as Rating Prediction and Evaulation.

Recommender systems are algorithms aimed at suggesting relevant items to users. They are critical in some industries as they can generate a huge amount of income when they are efficient. So in addition to our sentiment analysis, we also want to build a recommender system. We plan to do this by utilizing a few different approaches.

First is collaborative filtering. Collaborative filtering captures the correlation between user preferences and item features, so we can determine items that a user might like on the basis of reactions by similar users. This can be accomplished by using the Pearson correlation as it will help us to determine the similarity between users and items for our real-valued score predictions. Then, additional clustering will help us to find natural patterns and similarities between the similar items and users. Additionally, we want to utilize latent factor models, with dimensionality reduction, in order to make real valued score predictions for items. Last, we can predict potential ratings for user-product pairs that do not exist in the dataset. Collectivly, this process will create a recommender system for our Amazon Fine Food Marketplace.

### Conclusions

Sentiment Analysis, Topic Modeling, and Recommendation Systems can enhance the marketplace experience for both consumers & retailers.

In order to retain loyal customers, it is important to provide the best service possible. This can be accomplished with topic modeling and sentiment analysis. With this information, eCommerce can efficiently maximize user satisfaction by prioritizing product updates that will have the greatest positive impact. It has also been observed that the more engaging a website is, the more people will shop there. This will eventually increase the revenue of the eCommerce company. Recommender systems can help retailers keep their customers engaged.

Overall, these machine learning techniques can add value to marketplace businesses. This is accomplished through product reviews that create an improved customer service experience as well as good recommendations for the customer, which will give customers a better experience but will also help companies to sell more products. As a result, both retailers and consumers benefit.

### *Next Steps*

* Obtain more feature data for our users & products to expand our models
  * For instance, we could utilize web scraping of metadata for the product by using the ProductId in the dataset. This would produce more useful features like the sub-category or the price of the product that would allow us to do even more analysis with our models.
  * More features can provide more detailed information about the users and/or the products, which can allow for a more optimized hybrid recommendation system. This approach utilizes both collaborative filtering and latent factorization in order to produce the recommendations.
 
* Integrate Topic Modeling and Sentiment Analysis results with the Recommender System
  * We can combine each of the methods by applying the recommender system methonds on the predicted outputs from the sentiment analysis in order to make recommendations based on the reviews rather than just the raw score in order to capture more nuanced opinions.
 


