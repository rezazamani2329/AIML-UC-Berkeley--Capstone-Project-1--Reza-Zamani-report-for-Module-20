# Big Picture of the Project and its process:  
    


### 1- Using AIML to evaluate marketing strategy

 Suppose you take part in a meeting to evaluate the probability of defualt payment by customers 
 
 Speaker represents the strategy and now is your turn to give suggestion   
 
  You can suggest this: Oh, let see what AI says 


### 2- Structure designer of the project and Python code writer 

 Here is the answer of AIML for your suggestion, please follow it. You will find some good ideas

 This code has user friendly structure. It is classification issue in machine learning, as capstone project  for `UC Berkeley Professional Cerfiticate in AIML`, from `Berkeley Engineering` and `Berkeley HAAS`, and written by `Reza Zamani`

### 3- Process of the project

 **3-1- Business Understanding (Problem statment, purposes, questions, methodology, steps, phases of project)** 

 **3-2- Data understanding**  

 **3-3- Baseline and simple Models (to get general clue)**

 **3-4- Other models (in default strucutre) and model comparison (evaluation, choosing the best model** 

 **3-5- Model improvement (more feature engineering and model tuning)**  

 **3-6- Results of analyzing the default and suggestions**  


## **4- Stages of Poject** 

### **1-First Phase:**

**Data Understanding** 

- **Gerneral checking: missing data, duplicate, incorrect data, ...**
  
-  **Target Variable Visualization**
  
- **Numerical Features Visualization**
  
- **Catergorical Features Visualization**
  
- **Relationship between features, and thier relationship with target**
  
- **Correlation problem**

**Feature Engineering**  

**Defining Basemodel**

to check the accuracy of model, before using machine learning algorithms

**Defining the simple model**

to check the first effect of mahcine learnign algorithms in accuracy and other indices imporvement

**Using other algorithms, before hyperparameter tuning**

To improve the performance 

**Choosing the best algorithm with different criteria** 

### **2-Second Phase:** 

-  **Improving the quality of data with more engineering of features**

-  **Tuning the hyperparmeters**

-  **Choosing best paramters for each algorithm** 

- **Evaluating the algorithms with different criteria (accuracy, time, f1, percision, recall, roc)**  

-  **Choosing the best algorithm with different criteria**
  

###  **3-Third Phase:**

- **Feature Importance, SHAP, and deployment**
  
- **Policy recommendation** 


# 1- Business Understanding 

###   **1-1-Main Research Question**  
-  **Can Machine Learning models accurately predict whether credit card customers will default or not ?**

###  **1-2-Other Research Questions**   
- **What are the main factors affecting the probability of default by customers?**

- **what are the relative importance of factors affecting the probability of default by customers?**

###  **1-3-Problem Statement** 
Customers and also managers of financial institutions- a credit card financial institutions - want to have a better understanding of the factors that impact credit card payment default and get a solution in place that can predict with high accuracy whether customers will default on payments.

To solve this problem we are given a dataset that contains credit card payment information and it is our task to determine what factors impact payment default as well as build a model or an ensemble of models that can predict default.

From the standpoint of risk management, the predictive accuracy of the predicted chance of default will be more valuable than the binary outcome of classification - credible or not credible clients. We must determine which clients will fall behind on their credit card payments. Financial dangers are demonstrating a trend regarding commercial bank credit risk as the financial industry has improved dramatically.As a result, one of the most serious risks to commercial banks is the risk prediction of credit clients. The current project is being created in order to analyse and predict the above-mentioned database. This research aims to identify credit card consumers who are more likely to default in the next month.

###  **1-4-Main Goal** 
- **Using AIML models, predic whether credit card customers will default or not**

###  **1-5-Other Goals** 
- **Determine main features hasve ther highest impact on credit card dafault**

- **Understand the relationship between features together**

- **Understand the mechanism through which features affect default payment as target variabel**

###  **1-6- Methodology** 
- **Classification**

###  **1-9-Methods (AIML algorithms)**

- 1- **Dummy Classifier**

- 2- **Logistic Regression**

- 3- **K-Nearest Neigbors (KNN)**

- 4- **Decision Trees**

- 5- **Support Vector Machine (SVM)**

- 6- **Random Forest**

- 7- **AdaBoost**

- 8- **XGBoost**

- 9- **Naive Bayes**

- 10- **Multi Layer Perceptron (MLP)**

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

From April to September of 2005, we tracked historical monthly payment records.The payback status is measured using the following scale: -2=no spending, -1=paid in full, and 0=use of revolving credit (paid minimum only).

1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.

* **PAY_0:** Repayment status in September, 2005

* **PAY_2:** Repayment status in August, 2005

* **PAY_3:** Repayment status in July, 2005

* **PAY_4:** Repayment status in June, 2005

* **PAY_5:** Repayment status in May, 2005

* **PAY_6:** Repayment status in April, 2005

**Amount of bill statement (NT dollar).**
* **BILL_AMT1:** Amount of bill statement in September, 2005

* **BILL_AMT2:** Amount of bill statement in August, 2005

* **BILL_AMT3:** Amount of bill statement in July, 2005

* **BILL_AMT4:** Amount of bill statement in June, 2005

* **BILL_AMT5:** Amount of bill statement in May, 2005

* **BILL_AMT6:** Amount of bill statement in April, 2005

**Amount of previous payment (NT dollar).**
* **PAY_AMT1:** Amount of previous payment in September, 2005

* **PAY_AMT2:** Amount of previous payment in August, 2005

* **PAY_AMT3:** Amount of previous payment in July, 2005

* **PAY_AMT4:** Amount of previous payment in June, 2005

* **PAY_AMT5:** Amount of previous payment in May, 2005

* **PAY_AMT6:** Amount of previous payment in April, 2005
 

**Output variable (desired target):**  


* **default.payment.next.month:** Default payment (1=yes, 0=no)
 
###   **2-3- General check of data, info, missing and duplicate** 
-Missing data, outlier and buplicated data are checke. 
- **Handling Missing Values:**
   - `No missing values`. 
###   **2-4- checking Variables correctness** 
 **Data correcting:**
   - 9 Features have some incorrect data:  `EDCATION', 'SEX', 'MARRIAGE' and 'PAY_1' to 'PAY_6'
   - we correct mistakes
###  **2-5-Data Wrangling** 
   - rename most of features to be easily followed
   - change variables education, sex, and marriage from unmerical to categorical data to follow easily and make sense 
   - change target variale in this phase from numerical to categorical, later after data visuaization we will change it again to numerical
###  **2-6- Target variable Analysis** 
- we have imbalanced dataset, then our project in imbalanced dataset classification
###  **2-7- Numerical variables visulaization** <
  
- **Numerical features:**
- `LIMIT_BAL`,`AGE`,
- `'PAY_SEPT`, `PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR`,
- `BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'BILL_AMT_JUN', 'BILL_AMT_MAY','BILL_AMT_APR`,
- `'PAY_AMT_SEPT', 'PAY_AMT_AUG', 'PAY_AMT_JUL', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR'`,
- **using different methods for visualization:**
- `scatterplot, histplot, violinplot, catplot, pariplot, kdeplot, boxplot, barplot`,
- we got various general point for numerical features and their effect on target variable (default payment)

###  *2-7-1- Visulization of "LIMIT_BAL" feature (ballance limit)* 

###  *2-7-2- Visualization of "age" feature* 

###  *2-7-3- Relationship between age and other factors* 

### *2-7-4- Visualization of history of payments* 

### 2-7-4-1- Probability of default with attention to history of payments 

### *2-7-4-2- pair plot to have big picture*   

### *2-7-5- Visualization of Monthly Bill Amonut* 

#### *2-7-5-1- Distribution of Monthly Bill Amonut* 

####  *2-7-5-3- pariplot of Bill Amonut to have big picture and relationship betwen them* 

###  *2-7-6- Visualization of Previous payment*  

#### *2-7-6-1- Distribution of pervious payment  Bill Amonut* 

#### *2-7-6-3- pariplot of previous payment to have big picture and relationship betwen them* 

###  *2-7-6-4- Relationship between bill amount and previous payment in each month*  

###  *2-7-7- Correlation between  numerical features* 
- we find some features have correlation with each other
- using VIF, we find that `BILL_AMT_AUG` has the maximum correlation 
###  **2-7-7-1-  using **VIF** to check the correlation conditions**
### **2-8-Categorical variable Analysis**
- Categorical features are:`EDUCATION`, `SEX`, and `MARRIAGE`
- For categorical data we check distribution of each feature (with counplot), their effect on target variable with heatmap, and their relationship with defualt payment.  
###  *2-8-1- Visulization of "gender" feature* 

###  *2-8-2- Visulization of "education" feature* 

### *2-8-3- Visualization of "Marriage" feature*  

#### *2-8-4- Heatmap for categorical variables* 

### *2-8-5- Probability of default with attention to categorical variables*

### *2-8-6- Relationship between balance limit, education, gender, and marriage* 

# **3- Engineering Features** 
- 1- check the dataset info to be sure have all data or do not have extra 

- 2- `Remove Highly Correlated feaute`

- 3- Split the data into features and target

- 4- Identify `categorical and numerical columns`

- 5- `Encode the target variable`

- 6- `Onehotencoder` fo catgorical and `standardscaler` for numerical varibales 

#  **4- Handling Imbalanced Datasets and train/test split** 
Our target variable in imbalance, then we should use the approach attetion to this.

There are different approaches to split imbalace classification. here are some of them 

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

##   **4-1-SMOTE method for Imbalanced Dataset** 

##  **4-2-Train/Test Split** 

#   **5- A Baseline Model**  

##  **5-1- Dummy Classifier** 
- **Dummy Classifier Accuracy: 0.5**


Classification Report:
                precision    recall  f1-score   support

           0       0.50      1.00      0.67      4664
           1       0.00      0.00      0.00      4682

    accuracy                           0.50      9346
   macro avg       0.25      0.50      0.33      9346
weighted avg       0.25      0.50      0.33      9346

#  **6- A Simple Model** 

###  **6-1-Logistic Regression** 

Logistic Regression Accuracy: 0.62


Classification Report:
                precision    recall  f1-score   support

           0       0.65      0.51      0.57      4664
           1       0.60      0.73      0.66      4682

    accuracy                           0.62      9346
   macro avg       0.63      0.62      0.61      9346
weighted avg       0.62      0.62      0.61      9346

####  **Point: Score of Simple model (0.62) is higher than Baseline model (0.50)** 

####  **Point: which shows, using machine learning algorihtms will improve the accuracy of model** 

#  **7- More Models to imporve performance** 

#### In this step, we use all modeLs **with default strucure**

####  In next stepts we will tune the moldes to find best parameters 

- 3- **K-Nearest Neigbors (KNN)**

- 4- **Decision Trees**

- 5- **Support Vector Machine (SVM)**

- 6- **Random Forest**

- 7- **AdaBoost**

- 8- **XGBoost**

- 9- **Naive Bayes**

- 10- **Multi Layer Perceptron (MLP)**

#   **8-Best Model** 

###   **in this step, best model is chosen before tuning the hyperparameters** </span>

### Evaluaion: 

 	`                   ` Train Score 	Test Score 	Average Fit Time 	Precision 	Recal 	f1
  
Model 		

Decision Tree 	`         `1.00 	`    `0.78 	`      `1.19 	`         `0.77 `   `0.79 `  `	0.78

KNN (k-Nearest Neighbors) `   `	0.86 `   ` 	0.78 `     ` 0.02 `      `  0.73`      ` 	0.89`   ` 	0.80

SVM (Support Vector Machine)`    ` 	0.73 `    `	0.71 `     `127.37 `     ` 	0.76 `    `	0.62 `    ` 	0.68

Logistic Regression `    ` 	0.63 `     `	0.62 `     ` 	0.33 `    ` 	0.60 `    ` 	0.73 `    ` 	0.66

Random Forest `         `	1.00 `    `	0.87 `     ` 	16.52 `     ` 	0.89 `       ` 	0.84 	`     `0.86

AdaBoost 	`         `0.74 `      ` 	0.73 `        ` 	6.15 `   ` 	0.78 `        ` 	0.65 `      ` 	0.71

XGBoost 	`             `0.91 `      `	0.86 `          `	1.24 `         `	0.91 `        `	0.80 `       `	0.85

Naive Bayes 	`           ` 0.56 `         ` 	0.55 `            ` 	0.03  `        `	0.53 `    `	0.92 `    `	0.67

MLP (Multi Layer Perceptron) 	`        `0.77 	`       ` 0.73 `             ` 	54.07`  ` 	0.71 `      `  	0.79 `  ` 	0.75



##  **8-1-General findings** 
- 1- Maximum Test score is 0.87

- 2- Maximum Train score is 1

- 3- Maximum f1 score is 0.86

- 4- Maximum Recal is 0.92

- 5- Maximum Precision is 0.91

- 6-minimum test and train score belongs to Naive Bayes, however it is higher than dummyclassifier (score =0.50) 

- 7- Generally, scores are very good and then the resutls are reliable

##  **8-2- comparison with attention to train and test score** 
1- **Train Score 	Test Score**: 

- Random Forest (1), 

- Decision Tree(1) 

- and XGBoost (0.91) have the highest score 

2- **Test Score**: 

- Random Forest (0.87), 

- XGBoost (0.86),

- Decision Tree (0.78) and KNN (0.78) have the highest score

##  **8-3- comparison with attention to precision, recall and f1:** 

1- **f1**: 

- Random Forest (0.86),

- XGBoost (0.85), 

- and KNN (0.80) have the highest score 

2- **Recall**: 

- Naive Bayes (0.92),

- KNN (0.89),

- and Random Forest (0.84) have the highest score 

3- **Precision**: 

- XGBoost (0.91), 

- Random Forest (0.89) 

- and Decision Tree (0.77) have the highest score 


##  **8-4- comparison with attention to Average Fit Time** 
- 1- KNN, Naive Bayes and Logistic Regression have the minimum Average Fit Time 

- 2- SVM and MLP have the maximum Average Fit Time


##  **8-5- choosing best models with attention to score, f1, recall, precision, and time together** 

### **Best model** ( before tuning the hyperparameters): **`Random Forest`**   
### **Second best model**:  `XGBoost`  
### **third best model**: `Decision Tree`.  

# **Next stpes in finall version of capstone project we will complete following steps** 

###  **1- More feature engineering to improve the quality of data** 

###  **2- Use different models, from normal models to ensemble models and neural networks**  

### **3- Tune the hyperparamteters in each model and choose best parameters** 
###  **4- ROC curve and confusion matrix**  

###  **5- Feature permutation, feature importance and SHAP** 
###  **6-conclusion and summary** 
###  **7-Policy recommendations and deployments** 


 
