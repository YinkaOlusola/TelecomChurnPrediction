# **CUSTOMER CHURN PREDICTION**

# **Introduction**

In the dynamic landscape of business, customer churn, or the departure of customers from a product or service, is a critical metric that significantly impacts the success or failure of a company.

The telecommunications industry, serving as a cornerstone for connectivity services, faces the challenge of predicting and minimizing customer churn. The churn rate, measured over a specific time frame, reflects the percentage of customers discontinuing services, highlighting the need for effective customer retention strategies.

<br>

<img src="images/customer_churn.png" alt="Image Description" width ="800" height="500">

<br>

Customer churn is particularly pertinent in industries with multiple options for consumers, where dissatisfaction or difficulties can prompt users to explore alternatives. For businesses, the cost of acquiring new customers surpasses that of retaining existing ones. Successful customer retention not only increases the average lifetime value but also enhances the sustainability and growth potential of a company. In this context, customer churn prediction emerges as a crucial task, allowing organizations to proactively address potential issues, implement tailored retention strategies, and maximize the value of recurring subscriptions.

This task focuses on leveraging machine learning techniques to analyze the Telco customer churn dataset, emphasizing the importance of predicting and mitigating customer churn for sustained business success [1], [2], [3] [4].


<br>

## **Dataset Overview**

The Telco customer churn dataset provides insights into the interactions of a fictional telecommunications company with 7043 customers in California. Each entry in the dataset encompasses diverse demographic information, including customer tenure, contract details, internet service specifics, and additional features. Of particular interest are the target variables - the Satisfaction Score, Churn Score, and Customer Lifetime Value (CLTV) index, which collectively contribute to understanding customer loyalty and engagement.

The following are the columns of the dataset:

  - `CustomerID`: A unique ID that identifies each customer.
  - `Gender`: The customer’s gender: Male, Female
  - `SeniorCitizen`: Indicates if the customer is 65 or older: 1 (Yes), 0 (No).
  - `Partner`: Indicates if the customer is married: Yes, No
  - `Dependents`: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
  - `Tenure`: Indicates the total amount of months that the customer has been with the company.
  - `PhoneService`: Indicates if the customer subscribes to home phone service with the company: Yes, No
  - `MultipleLines`: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
  - `InternetService`: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
  - `OnlineSecurity`: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
  - `OnlineBackup`: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
  - `DeviceProtection`: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
  - `TechSupport`: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
  - `StreamingTV`: Indicates if the customer uses their Internet service to stream television programme from a third party provider: Yes, No. The company does not charge an additional fee for this service.
  - `StreamingMovies`: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
  - `Contract`: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
  - `PaperlessBilling`: Indicates if the customer has chosen paperless billing: Yes, No
  - `PaymentMethod`: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
  - `MonthlyCharge`: Indicates the customer’s current total monthly charge for all their services from the company.
  - `TotalCharges`: Indicates the customer’s total charges, calculated to the end of the quarter specified above.
  - `Churn`: Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value. [6].

<br>


## **Problem Statement**
The Telco Churn Prediction task aims to address the challenge of developing an effective predictive model to anticipate and manage customer attrition within the telecommunications industry.

## **Objective**
Develop a predictive model to identify customers at risk of churning based on historical data and relevant features.

The developed model should provide telecom companies with a proactive tool to identify customers at risk of churning, enabling them to implement targeted retention strategies and enhance overall customer satisfaction. The ultimate goal is to reduce customer churn, thereby contributing to increased customer lifetime value and the overall success of the telecommunications business.


## **Algorithm Selection**

The Telco Churn Prediction task involves classifying customers into two categories: those likely to churn and those likely to stay.

This task is framed as a binary classification problem, where the algorithm must learn patterns and relationships within the dataset to predict whether a customer is part of the "Churn" class or the "No Churn" class.

Logistic Regression is a suitable choice for this task due to its interpretability. The telecom industry often requires explanations for business decisions, and Logistic Regression provides coefficients for each feature, making it easier to understand the impact of predictors on the likelihood of churn.



# **Exploratory Data Analysis**

### **Data Visualization**

As the data types of most of the features in the dataset are categorical, a count plot will be used to view the categorical features based on the churn rate [2].

#### **Bar Plots for Categorical Features**


<img src="images/plot_gender.png" alt="Image Description" width ="800" height="300">

<br>

Gender and partner distribution in the dataset is relatively balanced, with approximately equal values for each category. Although there is a slightly elevated churn rate in females, the marginal difference is considered negligible. 

<br>

<br>

<img src="images/plot_SeniorCitizen.png" alt="Image Description" width ="800" height="300">

<br>

<br>

<br>

<img src="images/plot_Partner.png" alt="Image Description" width ="800" height="300">

<br>

<br>

<br>

<img src="images/plot_Dependents.png" alt="Image Description" width ="800" height="300">

<br>

Notably, as can be seen from the three plots above a discernible proportion of churn is observed in younger customers (SeniorCitizen = 0), customers without partners, and those without dependents. The demographic analysis underscores non-senior citizens without partners and dependents as a specific customer segment exhibiting a higher likelihood of churning. 

<br>

<br>

<img src="images/plot_PhoneService.png" alt="Image Description" width ="800" height="300">

<br>

<br>

<br>

<img src="images/plot_MultipleLines.png" alt="Image Description" width ="800" height="300">

<br>


If a customer lacks phone service, the possibility of having multiple lines is excluded. Notably, a considerable proportion of customers subscribing to phone services exhibits a heightened likelihood of churning.

<br>

<br>

<img src="images/plot_InternetService.png" alt="Image Description" width ="800" height="300">

<br>

 Conversely, customers with fiber optic as their internet service demonstrate an increased propensity to churn, potentially influenced by factors such as elevated costs, market competition, and customer service concerns. The comparatively higher cost of fiber optic service, as opposed to DSL, could be a contributing factor to customer attrition.

<br>

<br>

<img src="images/plot_OnlineSecurity.png" alt="Image Description" width ="800" height="300">

<br>

<br>

<br>


<img src="images/plot_OnlineBackup.png" alt="Image Description" width ="800" height="300">

<br>

<br>

<br>


<img src="images/plot_DeviceProtection.png" alt="Image Description" width ="800" height="300">

<br>

<br>

<br>

<img src="images/plot_TechSupport.png" alt="Image Description" width ="800" height="300">

<br>

Customers availing services like OnlineSecurity, OnlineBackup, DeviceProtection, and TechSupport are notably less inclined to churn.

<br>

<br>

<img src="images/plot_StreamingTV.png" alt="Image Description" width ="800" height="300">

<br>

<br>

<br>

<img src="images/plot_StreamingMovies.png" alt="Image Description" width ="800" height="300">

<br>

Interestingly, the availability of Streaming service appears to exhibit no discernible predictive pattern for churn, as it is evenly distributed among both 'yes' and 'no' options.

<br>

<br>

<img src="images/plot_Contract.png" alt="Image Description" width ="800" height="300">

<br>

The shorter the contract, the higher the churn rate. Those with more extended plans face additional barriers when canceling early. This clearly explains the motivation for companies to have long-term relationships with their customers.

<br>

<br>

<img src="images/plot_PaperlessBilling.png" alt="Image Description" width ="800" height="300">

<br>

<br>

<img src="images/plot_PaymentMethod.png" alt="Image Description" width ="800" height="300">

<br>

 Churn Rate is higher for the customers who opted for paperless billing. Customers who pay with electronic checks are more likely to churn, and this kind of payment is more common than other payment types.

<br>


In general, the plots above show a consistently low churn rate for each categorical features. Moreso, from the plots, there is a trace of data imbalance which will be investigated using charts.

<br>

#### **Visualizing the Churn rate per category**

<br>

<img src="images/churn_rate.png" alt="Image Description" width ="800" height="300">

<br>

The plotted count distribution provides an overview of the `Churn` column in the dataset. The x-axis represents the churn status, distinguishing between customers who have churned and those who have not. The y-axis indicates the count of instances in each category.

This visualization is instrumental for assessing data balance, as an equitable distribution between the two classes is essential for training a robust machine learning model.

The height of the bars show that there is a significant imbalance in the dataset and it is essential to address these imbalances through techniques such resampling.


The dataset exhibits a notable class imbalance, with a substantial majority class and a smaller minority class. Given this imbalance, machine learning algorithms renowned for their ability to handle imbalanced datasets will be employed for model training.

Notably, `Random Forest` and `Naive Bayes Classifier` are selected, as these algorithms inherently account for class distribution in their learning processes. Random Forest, in particular, stands out as an ensemble method that has demonstrated effectiveness in addressing class imbalance challenges.

<br>

#### Distribution of Numerical Features

<img src="images/histograms_of_numerical_features.png" alt="Image Description" width ="800">

<br>

The Histogram above depicting customer tenure reveals a right-skewed distribution, indicating that a sigificant portion of customers has a relatively short association with the telecommunication company, predominantly within the initial 0-9 months. Correspondingly, the highest incidence of churn is observed during this initial period, highlighting that a considerable number of customers decide to discontinue services within their first 30 months.

A closer examination of the monthly charge histogram reveals a noteworthy pattern - customers with higher monthly charges display an increased churn rate. This finding suggests a potential correlation between the attractiveness of discounts and promotions and customer retention.

In essence, it implies that the provision of discounts or promotional offers could serve as a compelling incentive for customers to sustain their subscriptions, thereby reducing the likelihood of churn. This insight highlights the importance of promotional strategies in fostering customer loyalty and mitigating churn risks.


<br>

<img src="images/box_plots_of_numerical_features.png" alt="Image Description" width ="800">

<br>

The insights derived from the box plots above align with the findings from the histogram, reinforcing the observation about customer churn.

The combination of the Box plots and Histograms provides a comprehensive understanding of the distribution and tendencies within the numerical features of the telco dataset. The identified patterns, supported by the presence of outliers, contribute valuable insights into potential factors influencing customer churn, such as tenure duration, monthly charges and total expenditure.


### **SMOTE for Data Balancing**

In imbalanced datasets, where one class has significantly fewer instances than the other, machine learning models might be biased toward the majority class.

SMOTE - which stands for Synthetic Minority Over-sampling Technique - can be used to mitigate this issue by creating synthetic samples for the minority class, thus achieving a more balanced distribution.

<br>

<img src="images/balanced_data.png" alt="Image Description" width ="800">

<br>

SMOTE was applied to the dataset to balance the class distribution, then the models were trained on the balanced dataset and the performance of the models were observed and compared with the baseline model.

<br>


## RESULTS


| Model | Accuracy | Precision | Recall | F1 Score |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| Logistic Regression (Baseline model) | 0.751040 |	0.687117 |	0.600536 |	0.817846 |
| Naive Bayes (Baseline model) | 0.738369 |	0.435864 |	0.892761 |	0.684321 |
| Random Forest (Baseline model) | 0.688360 |	0.651685 |	0.466488 |	0.780642 |
| Tuned Logistic Regression | 0.861053 |	0.874140 |	0.847474 |	0.860865 |
| Tuned Naive Bayes | 0.797912 |	0.760494 |	0.880839 |	0.797510 |
| Tuned Random Forest | 0.857305 |	0.876877 |	0.835081 |	0.856968 |

<br>

### **Model Comparison**

<img src="images/model_comparison.png" alt="Image Description" width ="800">

<br>

The graph above illustrates noteworthy improvements in all performance metrics for the models following tuning, particularly when a balanced dataset was used.

Notably, the logistic regression model exhibited the most substantial improvement, ultimately achieving the highest F1 score among the tuned models.

This observation underscores the effectiveness of dataset balancing techniques in refining model performance and highlights the logistic regression model as particularly promising in the context of the analyzed metrics.

## FINDINGS

**Logistic Regression**

  - This model attained a **`balanced accuracy`** of **`86.25%`**, accompanied by an **`F1 score`** of **`86.23%`**, showcasing a well-balanced compromise between precision and recall. Its suitability for interpretability renders it adept at elucidating the influence of features on predictions.

 ***Advantages***

  Its strengths lie in its interpretability, simplicity, and efficient training speed.

 ***Drawbacks***

  However, there are limitations to consider, such as potential challenges in capturing intricate non-linear relationships within the data.

<br>

**Naive Bayes Classifier**

 - It attained a **`balanced accuracy`** registering at **`79.79%`** coupled with an **`F1 score`** of **`79.75%`**.

  ***Advantages***

  Naive Bayes is a simple and fast algorithm. It's computationally efficient and requires a relatively small amount of training data to estimate the parameters.

  Implementation of the Naive Bayes classifier is straightforward. It is easy to understand and doesn't involve complex parameter tuning.

  ***Drawbacks***

  Naive Bayes can be sensitive to the quality of the input data. If the training set is not representative of the real-world data, or if certain features have missing values, it may lead to biased predictions.


<br>

**Random Forest Classifier**

 - The model attained a **`balanced accuracy`** of **`85.97%`** and an **`F1 score`** of **`85.94%`**.

  ***Advantages***

  It demonstrates effectiveness in managing non-linear relationships and capturing intricate interactions within the dataset.

  The model is resilient against overfitting and offers insights into feature importance.

  ***Drawbacks***

  However, it necessitates a lengthier training period compared to Logistic Regression and might demand more meticulous hyperparameter tuning to achieve optimal performance.

<br>


# DISCUSSION

The evaluation of three distinct models revealed their effectiveness in predicting customer churn for telecom companies. Each model showcased commendable performance, with Random Forest excelling in capturing non-linear patterns, Logistic Regression providing a simpler, interpretable solution, and Naive Bayes demonstrating simplicity and faster training time.

<br>

**Strengths and Applications**

These predictive models offer invaluable insights, empowering telecom companies to proactively address customer churn through targeted retention strategies. The balanced accuracy and F1 scores indicate a harmonious trade-off between precision and recall, crucial metrics for evaluating model efficacy.

<br>

**Challenges and Considerations**

However, it's essential to acknowledge the impact of imbalanced datasets on model performance, particularly in terms of recall. To mitigate this issue, various strategies such as oversampling, undersampling, or alternative evaluation metrics were contemplated during the model tuning process.

<br>

**Recommendations for Model Enhancement**

To further optimize model performance, a focused approach on hyperparameter tuning, especially for Random Forest and Naive Bayes, is recommended.

Additionally, incorporating feature engineering techniques to generate more informative features has the potential to enhance overall model accuracy.

***Addressing Imbalanced Datasets***

Given the challenges posed by imbalanced datasets, exploring advanced techniques specifically designed to handle such scenarios is advisable.

In conclusion, the choice among these models should align with the unique needs and constraints of the telecom company, considering factors such as interpretability, computational efficiency, and the level of non-linearity in the data.



<br>

# CONCLUSION

The Telecommunication Churn Prediction has established a groundwork for proactive churn management within the telecommunications industry.

Although the showcased models demonstrated satisfactory performance, continuous improvement and the exploration of advanced techniques are essential to enhance the accuracy of predictions and provide superior decision support for telecom companies.

The iterative refinement process should remain integral, allowing for the adaptation to evolving data landscapes and ensuring the models stay at the forefront of churn prediction capabilities.

<br>

<br>

# References

[1] Introduction to Churn : https://www.kdnuggets.com/2019/05/churn-prediction-machine-learning.html

[2] Customer Churn and categorical plots: https://neptune.ai/blog/how-to-implement-customer-churn-prediction

[3] Churning : https://www.sciencedirect.com/science/article/pii/S2666603023000143

[4] Churn Rate : https://www.analyticsvidhya.com/blog/2022/09/bank-customer-churn-prediction-using-machine-learning/

[5] Dataset description : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

[6] Dataset overview : https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

[7] Encoding Categorical features : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder

[8] Precision as an Evaluation metric : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score

[9] Recall Score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score

[10] F1 Score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score