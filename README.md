# **CUSTOMER CHURN PREDICTION**

# **Introduction**

In the dynamic landscape of business, customer churn, or the departure of customers from a product or service, is a critical metric that significantly impacts the success or failure of a company.

The telecommunications industry, serving as a cornerstone for connectivity services, faces the challenge of predicting and minimizing customer churn. The churn rate, measured over a specific time frame, reflects the percentage of customers discontinuing services, highlighting the need for effective customer retention strategies.

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







# References

[1] Introduction to Churn : https://www.kdnuggets.com/2019/05/churn-prediction-machine-learning.html

[2] Customer Churn : https://neptune.ai/blog/how-to-implement-customer-churn-prediction

[3] Churning : https://www.sciencedirect.com/science/article/pii/S2666603023000143

[4] Churn Rate : https://www.analyticsvidhya.com/blog/2022/09/bank-customer-churn-prediction-using-machine-learning/