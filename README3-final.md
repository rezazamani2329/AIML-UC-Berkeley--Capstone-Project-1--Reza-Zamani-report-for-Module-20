# Big Picture of the Project and its process:  
    


### 1- Using AIML to evaluate marketing strategy

 Suppose you take part in a meeting to evaluate the probability of default payment by customers 
 
 Speaker represents the strategy and now is your turn to give suggestion   
 
  You can suggest this: Oh, let see what AI says 


### 2- Structure designer of the project and Python code writer 

 Here is the answer of AIML for your suggestion, please follow it. You will find some good ideas

 This code has user-friendly structure. It is classification issue in machine learning, as capstone project for `UC Berkeley Professional Certificate in AIML`, from `Berkeley Engineering` and `Berkeley HAAS`, and written by `Reza Zamani`

### 3- Process of the project
1- Business Understanding (Problem statement, purposes, questions, methodology, steps, phases of project)  </span>
#### <span style="color: Red; "> 2- Data understanding </span> 
#### <span style="color: Red; "> 3- Feature Engineering </span> 
####  <span style="color: Red; "> 4- Train/Test split and handling imbalanced dataset <span style="color: blue; ">  </span> 
####  <span style="color: Red; ">5- Machine learning models and their evaluation before hyperparamter tuning  </span> 
####  <span style="color: Red; ">6- Machine learning models after hyperparameter tuning  </span>
####  <span style="color: Red; ">7- Comparison of Models performance, evaluation, best model, and SHAP</span>
####  <span style="color: Red; ">8- Conclusion</span>

## **4- Stages of Project** 

#### <span style="color: blue; "> **1-First Phase:** </span> 

<span style="color: blue; "> **Data Understanding** </span> 
- **Gerneral checking: missing data, duplicate, incorrect data, ...**
-  **Target Variable Visualization**
- **Numerical Features Visualization**
- **Categorical Features Visualization**
- **Relationship between features, and their relationship with target**
- **Correlation problem**

<span style="color: blue; "> **Feature Engineering** </span> 

<span style="color: blue; "> **Using algorithms, before hyperparameter tuning** </span> 

<span style="color: blue; "> **Choosing the best algorithm with different criteria** </span> 

####  <span style="color: green; "> **2-Second Phase:** </span> 

- <span style="color: green; "> **Tuning the hyperparameters** </span> 

- <span style="color: green; "> **Choosing best parameters for each algorithm** </span> 

- <span style="color: green; "> **Evaluating the algorithms with different criteria (accuracy, time, f1, precision, recall, ROC_AUC)** </span> 

- <span style="color: green; "> **Choosing the best algorithm with different criteria** </span> 

#### <span style="color: orange; "> **3-Third Phase:** </span> 

- <span style="color: orange; "> **SHAP, and deployment** </span> 

- <span style="color: orange; "> **Policy recommendation** </span> 


# 1- Business Understanding 

###   **1-1-Main Research Question**  
-  **Can Machine Learning models accurately predict whether credit card customers will default or not?**

###  **1-2-Other Research Questions**   
- **What are the main factors affecting the probability of default by customers?**

- **what are the relative importance of factors affecting the probability of default by customers?**

###  **1-3-Problem Statement** 
Customers and also managers of financial institutions- a credit card financial institutions - want to have a better understanding of the factors that impact credit card payment default and get a solution in place that can predict with high accuracy whether customers will default on payments.

To solve this problem we use a dataset that contains credit card payment information and it is our task to determine what factors impact payment default as well as build a model or an ensemble of models that can predict default.

From the standpoint of risk management, the predictive accuracy of the predicted chance of default will be more valuable than the binary outcome of classification - credible or not credible clients. We must determine which clients will fall behind on their credit card payments. Financial dangers are demonstrating a trend regarding commercial bank credit risk as the financial industry has improved dramatically. As a result, one of the most serious risks to commercial banks is the risk prediction of credit clients. The current project is being created in order to analyses and predict the above-mentioned database. This research aims to identify credit card consumers who are more likely to default in the next month.

###  **1-4-Main Goal** 
- **Using AIML models, predict whether credit card customers will default or not**

###  **1-5-Other Goals** 
- **Determine main features have the highest impact on credit card default**

- **Understand the relationship between features together**

- **Understand the mechanism through which features affect default payment as target variable**

###  **1-6- Methodology** 
- **Classification**

###  **1-9-Methods (AIML algorithms)**

-  **Logistic Regression**

-  **K-Nearest Neighbors (KNN)**

- **Decision Trees**

- **Support Vector Machine (SVM)**

- **Random Forest**

- **AdaBoost**

- **XGBoost**

- **Naive Bayes**

- **Multi Layer Perceptron (MLP)**

#  **2- Data Understanding** 
##  **2-1-Load Dataset**  
##  **2-2- Understanding the Features** 

***Input variables:*** 

* **ID:** Unique ID of each client
* **LIMIT_BAL:** Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
* **Gender:** 1 = male; 2 = female
* **Education:** 1 = graduate school; 2 = university; 3 = high school; 4 = others
* **Marital status:** 1 = married; 2 = single; 3 = others.
* **Age:** Age in years

**History of past payment.**

From April to September of 2005, we tracked historical monthly payment records. The payback status is measured using the following scale: -2=no spending, -1=paid in full, and 0=use of revolving credit (paid minimum only).

1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.

* **PAY_0:** Repayment status in September 2005

* **PAY_2:** Repayment status in August 2005

* **PAY_3:** Repayment status in July 2005

* **PAY_4:** Repayment status in June 2005

* **PAY_5:** Repayment status in May 2005

* **PAY_6:** Repayment status in April 2005

**Amount of bill statement (NT dollar).**
* **BILL_AMT1:** Amount of bill statement in September 2005

* **BILL_AMT2:** Amount of bill statement in August 2005

* **BILL_AMT3:** Amount of bill statement in July 2005

* **BILL_AMT4:** Amount of bill statement in June 2005

* **BILL_AMT5:** Amount of bill statement in May 2005

* **BILL_AMT6:** Amount of bill statement in April 2005

**Amount of previous payment (NT dollar).**
* **PAY_AMT1:** Amount of previous payment in September 2005

* **PAY_AMT2:** Amount of previous payment in August 2005

* **PAY_AMT3:** Amount of previous payment in July 2005

* **PAY_AMT4:** Amount of previous payment in June 2005

* **PAY_AMT5:** Amount of previous payment in May 2005

* **PAY_AMT6:** Amount of previous payment in April 2005
 

**Output variable (desired target):**  

* **default.payment.next.month:** Default payment (1=yes, 0=no)
 
###   **2-3- General check of data, info, missing and duplicate** 
-Missing data, outlier and duplicated data are checked. 
- **Handling Missing Values:**
   - `No missing values`. 
###   **2-4- checking Variables correctness** 
 **Data correcting:**
   - 9 Features have some incorrect data:  `EDCATION', 'SEX', 'MARRIAGE' and 'PAY_1' to 'PAY_6'
   - we correct mistakes
###  **2-5-Data Wrangling** 
   - rename most of features to be easily followed
   - change variables education, sex, and marriage from numerical to categorical data to follow easily and make sense 
   - change target variable in this phase from numerical to categorical, later after data visualization we will change it again to numerical
###  **2-6- Target variable Analysis** 
- we have imbalanced dataset, then our project in imbalanced dataset classification
###  **2-7- Numerical variables visualization** <
  
- **Numerical features:**
- `LIMIT_BAL`,`AGE`,
- `'PAY_SEPT`, `PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR`,
- `BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'BILL_AMT_JUN', 'BILL_AMT_MAY','BILL_AMT_APR`,
- `'PAY_AMT_SEPT', 'PAY_AMT_AUG', 'PAY_AMT_JUL', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR'`,
- **using different methods for visualization:**
- `scatterplot, histplot, violinplot, catplot, pariplot, kdeplot, boxplot, barplot`,
- we got various general point for numerical features and their effect on target variable (default payment)

###  *2-7-1- Visualization of "LIMIT_BAL" feature (balance limit)* 

###  *2-7-2- Visualization of "age" feature* 

###  *2-7-3- Relationship between age and other factors* 

### *2-7-4- Visualization of history of payments* 

### 2-7-4-1- Probability of default with attention to history of payments 

### *2-7-4-2- pair plot to have big picture*   

### *2-7-5- Visualization of Monthly Bill Amount* 

#### *2-7-5-1- Distribution of Monthly Bill Amount* 

####  *2-7-5-3- par plot of Bill Amount to have big picture and relationship between them* 

###  *2-7-6- Visualization of Previous payment*  

#### *2-7-6-1- Distribution of pervious payment Bill Amount* 

#### *2-7-6-3- pariplot of previous payment to have big picture and relationship between them* 

###  *2-7-6-4- Relationship between bill amount and previous payment in each month*  

###  *2-7-7- Correlation between  numerical features* 
- We find some features have correlation with each other
- using VIF, we find that four features (`BILL_AMT_AUG`, `BILL_AMT_MAY`, `BILL_AMT_JUL`, `BILL_AMT_JUN` have high correlation, then remove them from dataset 
###  **2-7-7-1-  using **VIF** to check the correlation conditions**
### **2-8-Categorical variable Analysis**
- Categorical features are: ‘EDUCATION`, `SEX`, and `MARRIAGE`
- For categorical data we check distribution of each feature (with counplot), their effect on target variable with heatmap, and their relationship with default payment.  
###  *2-8-1- Visualization of "gender" feature* 

###  *2-8-2- Visualization of "education" feature* 

### *2-8-3- Visualization of "Marriage" feature*  

#### *2-8-4- Heatmap for categorical variables* 

### *2-8-5- Probability of default with attention to categorical variables*

### *2-8-6- Relationship between balance limit, education, gender, and marriage* 

# **3- Engineering Features** 
- 1- check the dataset info to be sure have all data or do not have extra 

- 2- Split the data into features and target

- 3- Identify `categorical and numerical columns`

- 4- `Encode the target variable` with `LabdelEncoder()`

- 5- `OneHotEncoder` fo categorical and `StandardScaler()` for numerical variables 

#  **4- Train/test split and Handling Imbalanced Datasets** </span>  

Our target variable in imbalance, then we should use the approach attention to this.

There are different approaches to split imbalance classification. Here are some of them 

1. **Oversampling**
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- Random Oversampling (Simply duplicates minority class instances)
2. **Undersampling**
- Random Undersampling
- Tomek Links (Removes borderline samples)
- NearMiss (Keeps only the hardest-to-classify examples)
3. **Hybrid Methods**
- SMOTE + Tomek Links
- SMOTE + Edited Nearest Neighbors (ENN)
4. **Class Weighting**
5. **Algorithm-Level Methods**
6. **Data Augmentation**

##  **4-1-Train/Test Split** 
With data prepared, we split it into a train and test set.

##  <span style="color: red; "> **4-2-SMOTE method for Imbalanced Dataset** </span>  
<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
 
### <span style="color: blue; "> Points: Results of applying train/test spli and SMOTE </span> 
#### 1. <span style="color: red; ">  **At first we split data into train and test, then applied SMOTE, then model does not have data leakage**  </span>  
#### 2. <span style="color: red; ">  **Moreover, in this project, we find that if we apply SMOTE before splitting the data, we will have overfitting or underfitting, I Learn it from my supervisor in UC Berkeley**</span>  
#### 3. <span style="color: red; ">  **As we applying train/test at first, then applying SMOTE, we do not have data leakage and also check and find there is not any overfitting**</span>  

</div>

# <span style="color: red; ">  **5- Machine learning Models** </span> 
Strategy:

- 1-At first we define and fit each model
- 1- in each model, we see accuracy
- 2- in each model, we see classification report
- 3- check train score, test score, fitting time, precision, recall and f1
- 4- create dataframe to see all score together
- 5- with attention to all criteria, choose the best model in this step
- 6- we do not tune the models in this section, we will tune the models in next section
#### <span style="color: red; ">  **5-1-Decistion Tree** </span> 
#### <span style="color: red; ">  **5-2-XGBoost** </span> 
#### <span style="color: red; ">  **5-3-KNN (K-Nearest Neighbors)** </span> 
#### <span style="color: red; ">  **5-4-Logistic regression** </span> 
#### <span style="color: red; ">  **5-5-Random Forest** </span>
#### <span style="color: red; ">  **5-6-AdaBoost** </span> 
#### <span style="color: red; ">  **5-7-SVM (Support Vector Machine)** </span> 
#### <span style="color: red; ">  **5-8-MLP (Multi Layer Perceptron)** </span> 

Evaluation: 

 	`                             `Accuracy`        `Precision`       `Recall`          `f1`       `ROC_AUC`       `Average Fit Time
  
Model 		

Decision Tree`                      `0.78`            `0.50`            `0.48`           `0.49`       `0.72`           `0.53

XGBoost`                            `0.80`            `0.58`            `0.40`           `0.47`       `0.75`           `0.76

KNN (k-Nearest Neighbors)`          `0.68`            `0.36`            `0.57`           `0.45`       `0.70`           `0.01

Logistic Regression`                `0.70`            `0.39`            `0.62`           `0.48`       `0.73`           `1.69

Random Forest`                      `0.79`            `0.52`            `0.55`           `0.53`       `0.77`           `4.17

AdaBoost`                           `0.76`            `0.47`            `0.55`           `0.51`       `0.75`           `3.37

SVM (Support Vector Machine)`       `0.76`            `0.46`            `0.58`           `0.51`       `0.75`           `642.64

MLP (Multi Layer Perceptron)`       `0.72`            `0.41`            `0.60`           `0.49`       `0.74`           `46.64


## <span style="color: red; ">  **5-9-Best Model** </span> 
### <span style="color: green; ">  **in this step, best model is chosen before tuning the hyperparameters** </span>

1- **Accuracy**: 

- `XGBoost` has maximum score, followed by `random forest` and `decision trees`  

2- **Precision**: 

- `XGBoost` has maximum score, followed by `random forest` and `decision trees`  

3- **Recall**: 

- `Logistic regression` has maximum score, followed by `MLP` and `SVM`  

4- **f1**: 

- `Random forest` has maximum score, followed by `AdaBoost` and `MLP`

5- **ROC_AUC**: 

- `Random forest` has maximum score, followed by `AdaBoost`, `XGBoost` and `SVM`
  
6- **Fit Time**: 

- `KNN`, `Decision Tree`, and `XGBoost` have the minimum Average Fit Time 

- `SVM` and `MLP` have the maximum Average Fit Time
    
## <span style="color: green; "> **Best model before hyperparameter tuning**: <span style="color: red ; ">**Random Forest**   </span> 
- **It has maximum ROC_AUC**
- **It has the second maximum accuracy**
- **It has the maximum f1**
- **It has the maximum Recall**
- **It has also relatively low fit time**


#  <span style="color: red; "> **6- Models with hyperparameter tuning (best parameters)** </span>  

####  <span style="color: red; "> **6-1 Decision Tree (best parameters)** </span>  
#### <span style="color: red; ">  **6-2-KNN (best parameters)** </span> 
#### <span style="color: red; ">  **6-3-SVM (best parameters)** </span> 
#### <span style="color: red; ">  **6-4- logistic Regression( best parameters)** </span> 
#### <span style="color: red; ">  **6-5-Random Forest ( best parameters)** </span> 
#### <span style="color: red; ">  **6-6-AdaBoost (best parameters)** </span> 
#### <span style="color: red; ">  **6-7-XGBoost (best parameters)** </span> 
#### <span style="color: red; ">  **6-8-MLP (Multi Layer Perceptron) (best parameters)** </span> 

 **Here are the results:**
- 1- Best Params for Decision Tree classification is : {'model__criterion': 'log_loss', 'model__max_depth': 10, 'model__max_features': 'sqrt'}
  
- 2- Best Params for KNN classification is: {'model__n_neighbors': 2, 'model__p': 1, 'model__weights': 'distance'}

- 3- Best Params for SVM classification is : {'model__C': 1, 'model__kernel': 'rbf'}
  
- 4-Best Params for Logistic Regression classification is: {'model__C': 1, 'model__max_iter': 100, 'model__solver': 'liblinear'}
  
- 5- Best Params for Random Forest classification is : {'model__max_depth': 10, 'model__max_features': 'sqrt', 'model__n_estimators': 50}

- 6- Best Params for AdaBoost classification is : {'model__learning_rate': 1, 'model__n_estimators': 100}

- 7 - Best Params for XGBoost classification is: {'model__learning_rate': 0.1, 'model__max_depth': 10, 'model__n_estimators': 100}

-8-  Best Params for MLP classification is: {'model__activation': 'relu', 'model__hidden_layer_sizes': (5,)}



# <span style="color: red; ">  **7- Comparison of tuned models (with best parameters)** </span> 

#### <span style="color: red; ">  **7-1- Pipeline for best parameters for each model**  </span> 
#### <span style="color: red; ">  **7-2-Fit the models and make prediction**  </span> 
#### <span style="color: red; ">  **7-3- Evaluation** </span> 

`                                 `Accuracy`            `Precision`       `Recall`           `F1-Score`           `ROC-AUC
Model 

k-Nearest Neighbors (KNN)`        `0.858`               `0.810`           `0.939`             `0.870`               `0.890

Logistic Regression`              `0.628`               `0.611`           `0.730`             `0.666`               `0.677

Decision Tree`                    `0.742`               `0.767`           `0.703`             `0.734`	            `0.819

Support Vector Machine (SVM)`     `0.713`               `0.776`           `0.609`             `0.682`               `0.785

Random Forest`                    `0.795`               `0.842`           `0.733`             `0.783`               `0.876

AdaBoost`                         `0.748`               `0.805`           `0.665`             `0.728`               `0.824

XGBoost`                          `0.858`               `0.893`           `0.819`             `0.854`               `0.929

MLP (Multi Layer Perceptron)`     `0.702`               `0.747`           `0.621`             `0.678`               `0.774

#### <span style="color: red; ">  **7-4- plotting ROC curve** </span> 
#### <span style="color: red; ">  **7-5-  Confusion Matrix** </span> 
### <span style="color: red; ">  **7-6-Best Model after tuning the hyperparameters** </span> 

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
    
1- **Accuracy**: 

- `KNN`  and  `XGBoost`   have maximum  score, followed by `Random Forest`. 

2- **Precision**: 

- `XGBoost` has maximum score, followed by `random forest` and `KNN`  

3- **Recall**: 

- `KNN` has maximum score, followed by `XGBoost`  

4- **f1**: 

- `KNN` has maximum score, followed by `XGBoost`

5- **ROC_AUC**: 

- `XGBoost` has maximum score, followed by `KNN`.

  
### <span style="color: green; "> **Best model** after hyperparameter tuning: <span style="color: red ; ">**KNN**   </span> 
- **It has the maximum f1**
- **It has the maximum Recall**
- **It has the maximum accuracy**
- **It has the second maximum ROC_AUC**
- **It has the third maximum Precision**



### <span style="color: green; "> **Second Best model** after hyperparameter tuning: <span style="color: red ; ">**XGBoost**   </span> 
- **It has the maximum Precision**
-  **It has the maximum ROC_AUC**
- **It has the maximum accuracy**
- **It has the second maximum Recall**
- **It has the second maximum f1**

## <span style="color: red; ">  **7-7-feature importance and SHAP** </span> 
##### <span style="color: red; ">  **7-7-1-Feature Importance with the second best model (XGBoost)** </span> 
##### <span style="color: red; ">  **7-7-2-Feature Importance with the best model (KNN)** </span> 
##### <span style="color: red; ">  **7-7-3-SHAP  for the second best model (XGBoost)** </span> 
##### <span style="color: red; ">  **7-7-3-SHAP  for the best model (KNN)** </span> 

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 

# <span style="color: GREEN; "> **Main results from SHAP and feature importance:** </span>

## <span style="color: red; "> **1-3-Top Three features affecting the target:** <span style="color: GREEN; ">
## <span style="color: red; "> **AGE**, 
## <span style="color: red; "> **PAY_SEPT**,
## <span style="color: red; "> **LIMIT_BAL** 
##
### **Other findings:**
### 4- **History of payment**: it is generally important as its elements have relatively high raking in SHAP and **three among top five ranking** (`PAY_SEP`, `PAY_ARP`, `PAY_AUG` ) are among top five features
### 5- **PAY_AMT**: among different months, **April** has the **highest impact**, followed by **September**
### 6- **BILL-AMT**: bill amount of **September** has the **highest impact** among BILL_AMT, followed by April
### 7- **Among months**: **previous month is most important**. We have the feature for each six previous months (history of payment, last payment, and last bill). Among all six previous months, last month (in feature September) is most important, as PAY_AMT, BILL_AMNT_SEP, and PAY_AMNT_SEP have higher ranking from other months in similar situation.
### 8- **Among months**: **sixth previous month is the second most important**. By following all features for each month in SHAP (history of payment, last payment, last bill), we find that after September, April in the second important month. It shows that as financial institution, if you want to check the condition of customer, at first see the previous month, then go and check the six previous months, as it is more important the other months.
### 9- **SEX**: In SHAP, Male has higher ranking than Female, but has lower ranking in feature important. When we look at carefully to the colors and value in SHAP, we find that the probability of default by Female is lower which is compatible with data understanding we have and with feature importance
### 10- **MARRIAGE**: in SHAP (in both of them), we have only `MARRIAGE_Married', but looking at to feature importance, we have married and single with higher ranking for marries. We also find in data understanding that single has higher risk than married. 
### 11- **EDUCATION**: in SHAP, we have only `EDUCATION_University` as important factor affecting target. Feature important also shows that university degree has higher ranking, followed by graduate school and higher school. This finding in compatible with data understanding. Their combination shows that people with university degree and higher education have lower risk than other two groups. 
</div>


<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 

# <span style="color: GREEN; ">  **Interpretation** of results from SHAP and feature importance: </span>

## <span style="color: red; "> **1- PAY_SEPT** <span style="color: black; ">
#### 1-1- Last month is very important. If someone has delay in last month in payment or default, the probability of default in next month will be very high.
#### 1-2- Therefore there is maximum risk of default. As financial institution, if you see your customer has default in last month, try to call him/her to prevent from default in next month.
#### 1-3- Moreover as we find in data understanding section, if someone has default for two months, it would be possible to continue default or delay for all months.

## <span style="color: red; "> **2- LIMIT_BAL** <span style="color: GREEN; "> 
#### 2-1- it is an important feature affecting default.
#### 2-2- When this feature for a customer is increasing it means the customer is using more and more from credit card, which shows his debt are increasing. 
#### 2-3- Higher amount of credit card leads to higher risk of default. 
#### 2-4- As financial institution, when you see the trend of using credit card by customer and his/her family is increasing sharply, the probability of default is increasing too

## <span style="color: red; "> **3- AGE**  <span style="color: GREEN; ">
#### 3-1-Age is an important feature affecting the default payment
#### 3-2- Age group of 31-40 has the minimum probability of default, followed by age group of 21-30, 41-50 
#### 3-3- Age group of 71-80 has the maximum probability of default, followed by age group 61-70. 
#### 3-4 - As financial institution, you should more care from people who has more than 70 years of old.
#### 3-5- As financial institution, if you are conservative about default, you can more focus on people with age more than 60. 

### <span style="color: red; "> **customers with very high risk of default:**
#### Age group 71-80 
### <span style="color: red; "> **customers with high risk of default:**
#### Age group 61-70 
### <span style="color: red; ">**customers with low risk of default**: 
#### Age group 31-40 
### <span style="color: red; "> **Age with middle risk of default**: 
#### 21-30, 41-60 

### **Other findings:**

### 4- **History of payment**: 
#### Previous behavior of each customer can act as proxy for next behavior. Behavior of customer in last month (`PAY_SEP`) and in six month before (`PAY_ARP`) are very important. It means if someone started in the last month to default or from the first month of our data started to default, he/she has higher probability to default again.

### 5- **BILL-AMT**: 
#### Moreover, the bill amount of last month is important too, higher bill amount in last month has positive impact on default in coming month.  

### 6- **which month is important**:
#### We have three feature for each month, with attention to all information from SHAP, feature importance and data understanding:
#### - 6-1- **September** (last month)
#### - 6-2- **April** (first month)

#### This finding shows that if you want to check the behavior of customer check the first and last month behavior

### 7- **SEX**: 
#### Female has lower risk of default than Male, but it is not so important factor

### 8- **MARRIAGE**:
#### Married person has lower risk than single, but it is not so important factor 

### 9- **EDUCATION**: 
#### Customer with university degree has lowest risk, followed by higher education. It may be a proxy of income flow, and higher income flow can affect negatively the risk of default. Generally, education is normal factor and not important factor


# <span style="color: red; ">  **8- Conclusion** </span> 
## <span style="color: red; "> **8- 1- Steps of the project (what we have done?)** </span>

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
  
### 1- <span style="color: red; "> **Business understanding** </span>
    Purpose of this project is to predict the next behaver of customer that he or she will default or pay 
- 1-main question
- 2-other questions 
- 3-problem statement 
- 4-main goal
- 5- othe goals
- 6- methodology 
- 7- steps of project
- 8- methods 

#### 2- <span style="color: red; "> **Data understanding** </span> 
     Data understanding: In around 30k datasets (24 feature and 1 target),
    Target variable represents the default of payment.
- 1-load dataset 
- 2-undestanidng features 
- 3-general info, missing, duplicate 
- 4-checking correctness of variables and inside them 
- 5- data wrangling 
- 6-target variable analysis 
- 7- numerical variable visualization: visualization and correlation control  
- 8- categorical variable visualization 


### 3- <span style="color: red; "> **Feature Engineering** </span>
  - 1- check the dataset info to be sure have all data or do not have extra 

- 2- Remove Highly Correlated feature

- 3- Split the data into features and target

- 4- Identify categorical and numerical columns

- 5- Encode the target variable

- 6- Onehotencoder fo categorical and standardscaler for numerical variables 

### 4- <span style="color: red; "> **train/test split and handling imbalanced dataset** </span>

- 1-train/test split
- 2-SMOTE method for imbalanced dataset

### 5- <span style="color: red; "> **Machine learning models before hyperparameter tuning** </span> 
- 1- decision trees 
- 2- KNN
- 3- SVM
- 4- Logistic Regression 
- 5- Random Forest 
- 6- AdaBoost 
- 7- XGBoost 
- 8- MLP (multi layer perceptron)

- we use not only test and train score, but also recall, precision, f1, classification report and confusion matrix.
  
- Best model before hyperparameter tuning
        
- 1- general finding 
- 2- comparison of model in test and train score
- 3- comparison by f1, recall and precision 
- 4- Average time fit 
- 5- choosing the best model and second and third models
  
   **best model is Random Forest (best)**, followed by XGBoost, and Decision Trees 

### 6- <span style="color: red; "> **Models with hyperparameter tuning (best parameters)** </span>
    We define different parameters for our classifiers (8 classifiers) 
    and using gridsearch we find the best parameters for each classifier:
    
- 1- decision trees 
- 2- KNN
- 3- SVM
- 4- Logistic Regression 
- 5- Random Forest 
- 6- AdaBoost 
- 7- XGBoost 
- 8- MLP
  
### 7- <span style="color: red; "> **Comparison of tuned models** </span> 
     Again fit model with this classifiers with their best parameters and in    last step of modeling try to evaluate their performance.
    
- 1- pipeline for best parameters for each model
- 2- Fit the models and make prediction 
- 3- Accuracy and Classification report 
- 4- Plotting ROC curve 
- 5- Confusion Matrix
- 6- SHAP and feature importance 
  
 **Best model after tuning the hyperparameters** </span> 
     
- 1- General findings 
- 2- Best accuracy score
- 3- Best f1 score 
- 4- Best Recall score 
- 5- Best Precision score 
- 6- Best ROC-AUC
- 7- Choosing the best models

- Using confusion matrix, ROC curve, recall, precision and f1
- We find that among models with best parameters, **KNN** regression is the best.
  
- **KNN** followed by **XGBoost** are the best models 
  

 **Feature importance and SHAP** </span>  
 Most important features: 
- LIM_SEP
- BILL_AMT
- AGE 

## <span style="color: red; "> **8-2- Deployment** </span>
Now that we have settled on our models and findings, it is time to deliver the information to the client.  I am organizing my work as a basic report that details my primary findings.  Keep in mind that my audience are financial companies interested in fine-tuning their strategy in debt market and are sensitive to the default of payment by customers.

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 


## <span style="color: red; "> **1- Why we can trust to AIML model and its results and predictions**
 #### <span style="color: black; "> Results of AIMl models are reliable, as our best and second best models have higher accuracy, recall, precision, f1 and ROC_AUC. All criteria are more than 80%, which shows results are acceptable. 

## <span style="color: red; "> **2- What are the most important factors affecting the risk of default?** 
<span style="color: black; "> Three top factors are affecting the risk of default are
- `AGE`,
- `PAY_SEPT`,
- and `LIMIT_BAL`

## <span style="color: red; "> **3- How `AGE` is affecting the `risk of default`, can you devide people with age on high, low and middle risk of payment?**

- categorize people on decade into 21-30, 31-40, ...., 71-80

- the probability of default for each group is as following: 
- AGE_GROUP `       ` Risk of default
- 21-30`               `0.22
- 31-40`               `0.20
- 41-50`               `0.23
- 51-60`               `0.25
- 61-70`               `0.26
- 71-79`               `0.33
  
#### `Low risk` of default: `31-40`  age group
#### `High risk` of default: `71-80` age group
#### `Middle risk` of default: `51-60` age group
#### `Lower middle risk` of default: `21-30`, and `41-50` age groups
#### `Upper middle risk` of default: `61-70` age group

## <span style="color: red; "> **4- How `PAY_SEPT` (history of last month) is affecting the `risk of default`?** 
#### Last month is very important. If someone has delay in last month in payment or default, the probability of default in next month will be very high.
#### Therefore, there is maximum risk of default. As financial institution, if you see your customer has default in last month, try to call him/her to prevent from default in next month.
#### Moreover,  as we find in data understanding section, if someone has default for two months, it would be possible to continue default or delay for all months.

## <span style="color: red; "> **5- How `LIMIT_BAL` (using of credit card) is affecting the `risk of default`?** 
 
#### It is an important feature affecting default.
#### When this feature for a customer is increasing it means the customer is using more and more from credit card, which shows his debt are increasing. 
#### Higher amount of credit card leads to higher risk of default. 
#### As financial institution, when you see the trend of using credit card by customer and his/her family is increasing sharply, the probability of default is increasing too

## <span style="color: red; "> **6- How `History of payment` is affecting the `risk of default`?** 

#### Previous behavior of each customer can act as proxy for next behavior. Behavior of customer in last month (`PAY_SEP`) and in six month before (`PAY_ARP`) are very important. It means if someone started in the last month to default or from the first month of our data started to default, he/she has higher probability to default again.

## <span style="color: red; "> **7- How `BILL-AMT` is affecting the `risk of default`?** 
#### The bill amount of last month is important. Higher bill amount in last month has positive impact on default in coming month.  

## <span style="color: red; "> **8- Which `Month` has more valuable information about the `risk of default`?** 
#### We have three data for each month (history of payment, bill amount and pay amount). Generally two important months are: 
#### **`September`** (`last month`)
#### **`April`** (`first month`)
#### You should track this two months, they represents the behavior of customer
#### This finding shows that if you want to check the behavior of customer check the first and last month behavior

## <span style="color: red; "> **9- How `Gender (SEX) ` is affecting the `risk of default`?** 
#### Female has lower risk of default than Male, but it is not so important factor

## <span style="color: red; "> **10- How `MARRIAGE` is affecting the `risk of default`?** 
#### Married person has lower risk than single, but it is not so important factor 

## <span style="color: red; "> **11- How `EDUCATION` is affecting the `risk of default`?** 
#### Customer with university degree has lowest risk, followed by higher education. It may be a proxy of income flow, and higher income flow can affect negatively the risk of default. Generally, education is normal factor and not important factor



</div>



