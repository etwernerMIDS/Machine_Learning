# [Marketplace Insights](https://github.com/etwernerMIDS/Machine_Learning/tree/master/Projects/Final%20Project/marketplace_insights)

## Erin Werner

### [My Contribution](https://github.com/etwernerMIDS/Machine_Learning/tree/master/Projects/Final%20Project/marketplace_insights/rec_sys)

This was a group project where we collectivly created models that would benefit both retailers and consumers. My focus was on improving the consumer experience by building a recommender system that would suggest similiar products and users.

* **Network Analysis**

Because the data set was so large, I took a look at the networks for the users and products with some of the most reviews. This helped provide some high-level insight to the structure of the data. In the product network, each node is a product and each edge presents a user that has reviewed each product. In contrast, the user network is made up of user-nodes and product-edges. In each network, we can start to see some clustering among the nodes. This would indicate similarity amongst products or users. However, this is just a portion of the data, so we can expect to see the network fill out more with more data.

* **Collaborative Filtering**
  * *Pearson Correlation & Clustering* 

Collaborative filtering works around the interactions that users have with items. These interactions can help find patterns in the data that reveal information about the items or users. This technique is used to build recommenders that give suggestions to a user on the basis of the likes and dislikes of similar users. Collaborative filtering doesn’t require features about the items or users to be known. Instead, similarity is calculated only on the basis of the rating a user gives to an item. 

So, for the data, I decided to use the Pearson correlation method to determine similarity as it fit the Score-feature scale of 1-5. I was then able to do both user and product based collaborative filtering. To address the issue of complexity, I generated a notion of neighborhood. This subset the similar products/users to include only the set of (K) similar users for a particular user as well as the set of (K) similar products for a particular product. In this context, I set K to 50 so I would have 50 nearest neighbor for all the users/products.

* **Latent Factor Models**
  * *Singular Value Decomposition*
  
A different approach to building a recommender system involves a step to reduce or compress the large but sparse user-item matrix. If the matrix is mostly empty, reducing dimensions can improve the performance of the algorithm in terms of both space and time. As a result, the columns in the user matrix and the rows in the item matrix are called latent factors and are an indication of hidden characteristics about the users or the items. The factor matrices can provide such insights about users and items. This approach relies on features, rather than just ratings like collaborative filtering. In this context, I used 10 factors. However, there are not many categorical features in the data set, meaning latent factorization is not as useful in this context.

* **Score Prediction**

Last, I came up with a function that returns a score by taking user u and item i as the input parameters. The function outputs a score that quantifies how strongly a user u likes/prefers item i. This is done using the ratings of other people similar to the user. In this case, the score is equal to the sum of the ratings that each similar user gave to that item, subtracted by the average rating of that user, multiplied with the weight of how similar the user is. To measure the accuracy of the result, I used the Root Mean Square Error (RMSE). You can't fix particular threshold values for RMSE. So, compared the RMSE of both test and train datasets. 

*[Group Presentation Slides](https://docs.google.com/presentation/d/1ln0FKlLTTXQ1nCd-Q4Ot7ZrJhhrg0XdUfuA-YjJpk40/edit#slide=id.g8f23a824b8_0_1)*
