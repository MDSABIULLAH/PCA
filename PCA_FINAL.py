'''
1. Business Problem 
1.1.	What is the business objective?
1.2.	What are the constraints?
1.3.	Define success criteria

2. Work on each feature of the dataset to create a data dictionary as displayed in the below image:
 

3. Exploratory Data Analysis (EDA):
      3.1. Univariate analysis.
      3.2. Bivariate analysis.

4. Data Pre-processing 
4.1 Data Cleaning, Feature Engineering, etc.

5. Multivariate Analysis 
5.1 Build the model on the scaled data (try multiple options).
5.2 Perform the clustering and analyze the clusters.
5.3 Validate the clusters (try with the different numbers of clusters), label the clusters, and derive insights (compare the results from multiple approaches).
6. Use the clustered data and perform feature extraction using PCA and SVD. Compare the results.
7. Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
8. Deploy the best model using Python Flask on the local machine.
'''





'''

Problem Statements:
The average retention rate in the insurance industry is 84%, with the top-performing agencies in the 93% - 95% range. Retaining customers is all about the long-term relationship you build. Offering a discount on the client’s current policy will ensure he/she buys a new product or renews the current policy. Studying clients' purchasing behaviour to determine which products they're most likely to buy is essential. 
The insurance company wants to analyze their customer’s behaviour to strategies offers to increase customer loyalty.

CRISP-ML(Q) process model describes six phases:
1. Business and Data Understanding
2. Data Preparation
3. Model Building
4. Model Evaluation
5. Deployment
6. Monitoring and Maintenance

Objective: Maximize the Sales 
Constraints: Minimize the Customer Retention
Success Criteria: 
Business Success Criteria: Increase the Sales by 10% to 12% by targeting cross-selling opportunities on current customers.
ML Success Criteria: NA
Economic Success Criteria: The insurance company will see an increase in revenues by at least 8% 


'''






'''


Column	                       Data Type         Description
Customer	                   object	         Unique identifier for each customer.
State	                       object	         The state where the customer resides.
Customer Lifetime Value        float64           Total lifetime value of the customer, indicating the expected revenue from the customer over their entire relationship.
Response	                   object	         Whether the customer responded to an offer (Yes or No).
Coverage	                   object	         Type of insurance coverage the customer holds (e.g., Basic, Extended, Premium).
Education	                   object	         The highest level of education attained by the customer (e.g., High School, Bachelor, etc.).
Effective To Date	           object	         Date when the insurance policy became effective.
EmploymentStatus	           object	         The employment status of the customer (e.g., Employed, Unemployed, Retired, etc.).
Gender                         object	         The gender of the customer (Male or Female).
Income	                       int64	         Annual income of the customer (in USD).
Location Code	               object	         The type of residential area where the customer lives (e.g., Urban, Suburban, Rural).
Marital Status                 object	         The marital status of the customer (e.g., Single, Married, Divorced).
Monthly Premium Auto	       int64	         Monthly auto insurance premium paid by the customer (in USD).
Months Since Last Claim	       int64	         Number of months since the customer made their last insurance claim.
Months Since Policy Inception  int64	         Number of months since the customer's policy inception date.
Number of Open Complaints	   int64	         Number of complaints the customer has lodged.
Number of Policies	           int64	         Number of insurance policies the customer holds.
Policy Type	                   object	         Type of insurance policy (e.g., Personal Auto, Corporate Auto, Special Auto).
Policy	                       object	         The specific policy plan held by the customer (e.g., L1, L2, L3 for different policy levels).
Renew Offer Type	           object	         Type of renewal offer made to the customer (e.g., Offer 1, Offer 2, etc.).
Sales Channel	               object	         The channel through which the customer purchased the policy (e.g., Agent, Call Center, Web, etc.).
Total Claim Amount	           float64	         The total amount claimed by the customer (in USD).
Vehicle Class	               object	         The class of vehicle insured (e.g., Two-Door Car, Four-Door Car, SUV, etc.).
Vehicle Size	               object	         The size of the vehicle insured (e.g., Small, Medium, Large).



'''









# Importing necessary libraries
import pandas as pd
import sweetviz as sv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from feature_engine.outliers import Winsorizer
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn import metrics
import joblib
import pickle
from sqlalchemy import create_engine, text
from urllib.parse import quote
import dtale
from feature_engine.outliers import Winsorizer
from sklearn.cluster import DBSCAN, KMeans
from kneed import KneeLocator

# Reading the CSV file into a DataFrame
insurance = pd.read_csv("C:/Users/user/Desktop/DATA SCIENCE SUBMISSION/Data Science [ 4 ] - Dimension Reduction/PCA/AutoInsurance (1).csv")
insurance.head()  # Displaying the first few rows of the dataset for a quick look
insurance.info()  # Displaying info such as column data types and non-null values

# Setting up Database Connection
user = 'root'  # Database username
pw = '12345678'  # Database password
db = 'univ_db'  # Database name

# Creating an SQLAlchemy engine to connect to the MySQL database
engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

# Pushing the DataFrame 'insurance' to a SQL table named 'Auto_Insurances1'
insurance.to_sql('Auto_Insurances70', con=engine, if_exists='replace', chunksize=1000, index=False)

# Query to read the data back from the SQL table
sql = 'SELECT * FROM Auto_Insurances70;'
insurance_data = pd.read_sql_query(text(sql), engine)

# Displaying the first few rows and info about the SQL query result
insurance_data.head()
insurance_data.info()





# Automated Exploratory Data Analysis (EDA) using Sweetviz
my_report = sv.analyze(insurance_data)  # Analyzes the data
my_report.show_html('insurance_stats.html')  # Generates an HTML report with insights

# Interactive EDA using Dtale
my_report_dtale = dtale.show(insurance_data)  # Creates an interactive EDA dashboard
my_report_dtale.open_browser()  # Opens the Dtale interface in the browser





# AutoEDA report
"""
1. Numerical Summary:
    
Customer Lifetime Value:
Mean: 8,004.94
Range: 1,898.01 to 83,325.38

Income:
Mean: 37,657.38
Range: 0 to 99,981

Monthly Premium Auto:
Mean: 93.22
Range: 61 to 298

Months Since Last Claim:
Mean: 15.10
Range: 0 to 35

Months Since Policy Inception:
Mean: 48.06
Range: 0 to 99

Number of Open Complaints:
Mean: 0.38
Range: 0 to 5

Number of Policies:
Mean: 2.97
Range: 1 to 9

Total Claim Amount:
Mean: 434.09
Range: 0.10 to 2,893.24



2. Categorical Summary:
    
Top Categories:
State: Most frequent state is California.

Response: 7,877 customers responded with "No".

Coverage: Majority have Basic coverage.

Education: Most customers have a Bachelor degree.

EmploymentStatus: Most customers are Employed.

Gender: The distribution is roughly equal between M and F.

Vehicle Class: Majority have Four-Door Car.



3. Missing Values:
There are no missing values in the dataset. All columns have complete data.



4. Unique Values in Categorical Columns:
State: 5 unique states.

Response: 2 unique responses (Yes and No).

Coverage: 3 unique coverage types.

Education: 5 unique education levels.

EmploymentStatus: 5 unique statuses.

Vehicle Class: 6 unique vehicle classes.

Policy Type: 3 unique policy types.

Sales Channel: 4 unique channels.
"""




# Checking for null values in the dataset
insurance_data.isnull().sum()
# Results show no null values present

# Checking for duplicate records in the dataset
insurance_data.duplicated().sum()
# Results show no duplicate values present




'''
Feature Engineering steps:
1. High-Value Customer:
   - Business Goal: Retain high-value customers.
   - Feature: Identifies customers with high lifetime value 
2. Cross-Sell Potential:
   - Business Goal: Increase engagement and revenue through cross-selling.
   - Feature: Targets relevant customers for additional offers 
3. Engaged Customer:
   - Business Goal: Retain customers who engage with marketing.
   - Feature: Identifies customers who responded previously
4. Premium Affordability:
   - Business Goal: Improve retention by offering affordable products.
   - Feature: Determines if a customer is overpaying, guiding pricing strategies.
5. Risk Category:
   - Business Goal: Retain low-risk customers who are less likely to file claims.
   - Feature: Segments customers based on claim history.
6. Retention Risk:
   - Business Goal: Identify and retain at-risk customers.
   - Feature: Measures potential churn based on complaints.
7. Recency Since Last Policy:
   - Business Goal: Re-engage customers who haven’t interacted recently.
   - Feature: Identifies disengaged customers for re-engagement.
8. Potential Revenue Increase:
   - Business Goal: Drive revenue growth through cross-selling.
   - Feature: Focuses on customers with the highest cross-sell potential.
'''





'''
1. High-Value Customer:
Business Goal: Focus on retaining high-value customers, who contribute more to revenue.
Feature: High_Value_Customer identifies customers with high lifetime value, making them prime targets for retention and personalized offers.
2. Cross-Sell Potential:
Business Goal: Cross-selling to existing customers increases engagement and revenue.
Feature: Cross_Sell_Potential shows which customers are more likely to purchase additional products. This allows the company to target those customers with relevant offers.
3. Engaged Customer:
Business Goal: Loyal customers who engage with marketing efforts are more likely to stay.
Feature: Engaged_Customer identifies customers who have responded to offers in the past, allowing the company to prioritize them for future offers.
4. Premium Affordability:
Business Goal: Offering affordable products and discounts improves retention.
Feature: Premium_Affordability helps the company understand whether a customer is overpaying for their policies, enabling targeted pricing or discount strategies.
5. Risk Category:
Business Goal: Retain low-risk customers who are less likely to file claims.
Feature: Risk_Category segments customers based on their claim history, allowing the company to offer loyalty programs to low-risk customers or offer relevant insurance products.
6. Retention Risk:
Business Goal: Identify and retain at-risk customers.
Feature: Retention_Risk measures potential churn based on complaints. Customers with high complaint rates are more likely to leave, so retention efforts can be focused here.
7. Recency Since Last Policy:
Business Goal: Engage customers who haven’t interacted with the company recently to prevent churn.
Feature: Recency_Since_Last_Policy helps identify which customers may be disengaged, allowing the company to re-engage them with new offers or services.
8. Potential Revenue Increase:
Business Goal: Drive revenue growth by offering cross-sell opportunities to the right customers.
Feature: Potential_Revenue_Increase helps the company focus on customers who offer the most cross-sell opportunities, boosting revenue and loyalty.
'''





# Defining thresholds for the engineered features using manual inspection.
high_value_threshold = 10000  # Example threshold for high-value customers
premium_affordability_low = 50  # Ratio threshold for low premium affordability
premium_affordability_high = 150  # Ratio threshold for high premium affordability

# Aggregating data by Customer to compute relevant features
aggregated_data = insurance_data.groupby('Customer').agg({
    'Customer Lifetime Value': 'mean',
    'Total Claim Amount': 'sum',
    'Number of Policies': 'sum',
    'Monthly Premium Auto': 'mean',
    'Income': 'mean',
    'Response': lambda x: (x == 'Yes').sum(),  # Count positive responses
    'Number of Open Complaints': 'mean',
    'Months Since Last Claim': 'mean',
    'State': 'first',
    'Coverage': 'first',
    'Education': 'first',
    'EmploymentStatus': 'first',
    'Vehicle Class': 'first'
}).reset_index()


# # Aggregating data by Customer to compute relevant features
# aggregated_data = insurance_data.groupby('Customer').agg({
#     'Customer Lifetime Value': 'mean',
#     'Total Claim Amount': 'sum',
#     'Number of Policies': 'sum',
#     'Monthly Premium Auto': 'mean',
#     'Income': 'mean',
#     'Response': lambda x: (x == 'Yes').sum(),  # Counting how many times a customer has responded positively
#     'Number of Open Complaints': 'mean',
#     'Months Since Last Claim' : 'mean'
# }).reset_index()




# Renaming columns for clarity
aggregated_data.columns = [
    'Customer',                # Customer identifier
    'Avg_Lifetime_Value',      # Average Customer Lifetime Value
    'Total_Claim_Amount',      # Total Claim Amount
    'Total_Policies',          # Total Number of Policies
    'Avg_Monthly_Premium',     # Average Monthly Premium Auto
    'Avg_Income',              # Average Income
    'Total_Positive_Responses',# Total times customer has responded positively
    'Avg_Open_Complaints',     # Average Number of Open Complaints
    'Avg_Months_Since_Claim',  # Average Months Since Last Claim
    'State',                   # State - using the first occurrence
    'Coverage',                # Coverage - using the first occurrence
    'Education',               # Education - using the first occurrence
    'EmploymentStatus',        # Employment Status - using the first occurrence
    'Vehicle_Class'            # Vehicle Class - using the first occurrence
]


# # Renaming columns for clarity
# aggregated_data.columns = ['Customer', 'Avg_Lifetime_Value', 'Total_Claim_Amount', 'Total_Policies',
#                            'Avg_Monthly_Premium', 'Avg_Income', 'Total_Positive_Responses', 'Avg_Open_Complaints']





# Feature Engineering based on the business problem
aggregated_data['High_Value_Customer'] = (aggregated_data['Avg_Lifetime_Value'] > high_value_threshold).astype(int)
aggregated_data['Cross_Sell_Potential'] = (aggregated_data['Total_Policies'] > 2).astype(int)
aggregated_data['Engaged_Customer'] = (aggregated_data['Total_Positive_Responses'] > 0).astype(int)

# Premium affordability based on the ratio of income to monthly premium
aggregated_data['Premium_Affordability'] = pd.cut(
    aggregated_data['Avg_Income'] / aggregated_data['Avg_Monthly_Premium'],
    bins=[-1, premium_affordability_low, premium_affordability_high, float('inf')],
    labels=['Low', 'Medium', 'High']
)

# Risk category based on total claim amount
claim_amount_mean = aggregated_data['Total_Claim_Amount'].mean()
claim_amount_std = aggregated_data['Total_Claim_Amount'].std()

aggregated_data['Risk_Category'] = pd.cut(
    aggregated_data['Total_Claim_Amount'],
    bins=[-float('inf'), claim_amount_mean - claim_amount_std, claim_amount_mean + claim_amount_std, float('inf')],
    labels=['Low', 'Medium', 'High']
)

# Retention risk based on the number of complaints
aggregated_data['Retention_Risk'] = pd.cut(
    aggregated_data['Avg_Open_Complaints'],
    bins=[-float('inf'), 0.5, 2.5, float('inf')],
    labels=['Low', 'Medium', 'High']
)

# Recency since last policy (transforming months since policy inception)
aggregated_data['Recency_Since_Last_Policy'] = insurance_data.groupby('Customer')['Months Since Policy Inception'].max().reset_index(drop=True)

# Potential revenue increase (cross-sell potential * avg monthly premium)
aggregated_data['Potential_Revenue_Increase'] = aggregated_data['Cross_Sell_Potential'] * aggregated_data['Avg_Monthly_Premium']

# Displaying the final engineered dataset's information
aggregated_data.info()  # This provides a concise summary of the DataFrame, including column data types and non-null counts

# Displaying the first few rows of the final engineered dataset
aggregated_data.head()  # This displays the first few rows of the DataFrame to visually inspect the data







# Separate the numerical data from the DataFrame
numerical_data = aggregated_data.select_dtypes(include=['number', 'int64', 'float64'])

# Define a function to detect outliers using the Interquartile Range (IQR)
def detect_outliers_iqr_count(dataframe):
    # Dictionary to store the count of outliers per column
    outlier_counts = {}
    
    # Iterating over each column in the provided DataFrame
    for column in dataframe.columns:
        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define the bounds for identifying outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers by checking which data points are outside the bounds
        outlier_condition = (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
        
        # Count the number of outliers in the column
        outlier_count = outlier_condition.sum()
        
        # Store the outlier count in the dictionary
        outlier_counts[column] = outlier_count

    # Return the dictionary containing the count of outliers for each column
    return outlier_counts

# Identify outliers and obtain the counts for the numerical data
outlier_counts = detect_outliers_iqr_count(numerical_data)

# Print the count of outliers for each column
for col, count in outlier_counts.items():
    print(f"Outliers in {col}: {count}")


'''
# Dropping columns due to low variation detected during outlier treatment
# The Winsorization outliers technique was attempted, but these columns had very low variation.

# Encountered a ValueError due to low variation in the following columns:
# ['Total_Positive_Responses', 'Avg_Open_Complaints', 'High_Value_Customer', 'Engaged_Customer']

# These columns were problematic because the 'iqr' method relies on variation in the data,
# and the low variation caused the method to fail. 
'''


# Apply the Winsorizer to cap outliers in the numerical columns

# List of columns to be processed with Winsorizer
numerical_column_1 = [
    'Avg_Lifetime_Value', 'Total_Claim_Amount', 'Total_Policies',
    'Avg_Monthly_Premium', 'Avg_Income', 'Cross_Sell_Potential',
    'Recency_Since_Last_Policy', 'Potential_Revenue_Increase'
]

# Instantiate the Winsorizer with IQR method to cap both tails at 1.5*IQR
winsor = Winsorizer(
    capping_method='iqr',   # Using 'IQR' method for capping
    tail='both',            # Cap both tails (lower and upper)
    fold=1.5,               # 1.5 times the IQR for capping
    variables=numerical_column_1  # Specify columns to apply Winsorizing
)

# Fit the Winsorizer on the numerical data and transform it
numerical_data_winsorized = winsor.fit_transform(numerical_data)

# Display the first few rows of transformed numerical data to check the results
print(numerical_data_winsorized.head())

# Re-checking for outliers in the winsorized numerical data
outlier_counts1 = detect_outliers_iqr_count(numerical_data_winsorized)

# Print the count of outliers for each column after Winsorizing
for col, count in outlier_counts1.items():
    print(f"Outliers in {col}: {count}")

# Separate categorical data from the DataFrame
categorical_data = aggregated_data.select_dtypes(include=['object', 'category'])

# Print information and a preview of the separated data
print("Numerical Data:")
print(numerical_data.info())
print(numerical_data.head(), "\n")

print("Categorical Data:")
print(categorical_data.info())
print(categorical_data.head())

# Concatenate the processed numerical and categorical data
final_data_before_scaling = pd.concat([numerical_data_winsorized, categorical_data], axis=1)

# Display info about the concatenated, pre-scaled data
final_data_before_scaling.info()

# Prepare the data for further processing by dropping the 'Customer' column
final_data = final_data_before_scaling
final_data.drop(columns=['Customer'], inplace=True)

# Display info about the prepared data
final_data.info()

# Select numerical and categorical columns for pipeline creation
num_data = final_data.select_dtypes(include=['int64', 'float64', 'number'])
cat_data = final_data.select_dtypes(include=['object', 'category'])

# Create pipelines for processing both numerical and categorical data

# Pipeline to handle missing data and scale numerical columns
numerical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('scale', StandardScaler())                  # Scale data using Standard Scaler
])

# Pipeline


# Pipeline for encoding categorical data
categorical_pipeline = Pipeline(steps=[
    ('OnehotEncode', OneHotEncoder(sparse_output=False))  # One-hot encode categorical variables
])

# Using ColumnTransformer to apply specific transformations to different columns
preprocess_pipeline = ColumnTransformer(
    transformers=[
        ('numerical', numerical_pipeline, num_data.columns.values),  # Apply numerical_pipeline to numerical columns
        ('categorical', categorical_pipeline, cat_data.columns.values)  # Apply categorical_pipeline to categorical columns
    ],
    remainder='passthrough'  # Leave any remaining columns untransformed
)

# Fit the ColumnTransformer to the data
processed1 = preprocess_pipeline.fit(final_data)

# Save the preprocessing pipeline to a file using joblib
joblib.dump(processed1, 'processed1.joblib')     

# Display the current working directory to confirm where the file is saved
import os
os.getcwd()

# Transform the final_data using the fitted preprocessing pipeline
# Convert the transformed numpy array back to a DataFrame with the correct column names
insurance_clean = pd.DataFrame(processed1.transform(final_data), columns=processed1.get_feature_names_out())

# Display information about the transformed and cleaned data ready for clustering
insurance_clean.info()










# Perform PCA to reduce dimensionality
pca = PCA(n_components=16)  # Initialize PCA to reduce to 16 components

pca_fit = pca.fit(insurance_clean)  # Fit PCA on the cleaned data

import os
os.getcwd()  # Display current working directory

pca_res = pd.DataFrame(pca.transform(insurance_clean))  # Transform the data using fitted PCA and convert to DataFrame

pca_res  # Display the transformed PCA data

# Obtain PCA component weights
pca.components_

# Create a DataFrame of PCA components for closer inspection
components = pd.DataFrame(pca.components_).T

components  # Display the PCA components

print(pca.explained_variance_ratio_)  # Print the variance ratio explained by each component

# Calculate cumulative explained variance
var1 = np.cumsum(pca.explained_variance_ratio_)

print(var1)  # Print cumulative variance

# Evaluate PCA components using a KneeLocator
kl = KneeLocator(range(len(var1)), var1, curve='concave', direction="increasing")
kl.elbow  # Identify the "knee" or "elbow" in the plot, which suggests optimal number of components

# Plot the cumulative explained variance
plt.plot(range(len(var1)), var1)
plt.xticks(range(len(var1)))
plt.ylabel("Inertia")
plt.axvline(x=kl.elbow, color='r', label='axvline - full height', ls='--')
plt.show()

# Assuming pca_res is a DataFrame of PCA results
# Selecting the first 17 columns from the PCA results
pca_subset = pca_res.iloc[:, 0:17]

# Renaming the columns of the PCA subset
pca_subset.columns = ['pc0', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9',
                      'pc10', 'pc11', 'pc12', 'pc13', 'pc14', 'pc15']

# Optionally assign this processed subset back to 'final'
final = pca_subset
final.head()  # Display the first few rows of the PCA subset

joblib.dump(pca_fit, 'pca.joblib')  # Save the PCA model using joblib

import os
os.getcwd()  # Display current working directory

# KMeans Clustering

# List to store Total Within-Cluster Sum of Squares (TWSS)
TWSS = []
k = list(range(2, 9))  # Define range of cluster numbers to try

# Loop through each number of clusters to calculate and store TWSS
for i in k:
    kmeans = KMeans(n_clusters=i)  # Initialize KMeans with i clusters
    kmeans.fit(pca_res)  # Fit KMeans on the PCA-transformed data
    TWSS.append(kmeans.inertia_)  # Append the inertia (TWSS) to the list

# Create a scree plot to visualize the TWSS against number of clusters
plt.plot(k, TWSS, 'ro-')  # Plot TWSS against number of clusters
plt.xlabel("Number of Clusters")  # Label for x-axis
plt.ylabel("Total Within-Cluster Sum of Squares")  # Label for y-axis
plt.title("TWSS vs Number of Clusters")  # Title of the plot
plt.show()  # Show the plot

# Build KMeans clustering with a chosen number of clusters (e.g., k=4)
model = KMeans(n_clusters=4)  # Specify the number of clusters
model.fit(pca_res)  # Fit KMeans on the PCA-transformed data

# Displaying the cluster labels
cluster_labels = model.labels_  # Array of cluster labels assigned to each data point
print("Cluster Labels:", cluster_labels)  # Print the cluster labels

pickle.dump(model, open('model.pkl', 'wb'))  # Save the KMeans model using pickle

# Concatenate the original data with the cluster labels
final = pd.concat([final_data_before_scaling, pd.Series(cluster_labels)], axis=1)

final.head()  # Display the first few rows of the final DataFrame
final.info()

# Save the final DataFrame to a CSV file
final.to_csv('pca_cluster70.csv', encoding='utf-8', index=False)

import os
os.getcwd()  # Display current working directory















import pandas as pd

df = pd.read_csv('C:/Users/user/Desktop/DATA SCIENCE SUBMISSION/Data Science [ 4 ] - Dimension Reduction/PCA/pca_cluster70.csv')





# Rename the cluster column for clarity
df.rename(columns={'0': 'Cluster'}, inplace=True)

# Calculate the average income for each cluster
average_income_by_cluster = df.groupby('Cluster')['Avg_Income'].mean()

# Aggregate statistical values (mean, median, std, etc.) for each cluster
cluster_statistics = df.groupby('Cluster').agg({
    'Avg_Income': ['mean', 'median', 'std'],
    'Total_Claim_Amount': ['mean', 'median', 'std'],
    'Total_Policies': ['mean', 'median', 'std'],
    'Avg_Monthly_Premium': ['mean', 'median', 'std'],
    'Cross_Sell_Potential': ['mean', 'median', 'std'],
    'Potential_Revenue_Increase': ['mean', 'median', 'std']
})

# Display the results
print("Average Income by Cluster:")
print(average_income_by_cluster)
print("\nCluster Statistics:")
print(cluster_statistics)



# Mapping the cluster numbers to meaningful labels
cluster_labels = {
    0: "Low Lifetime Value, Basic Customers",
    1: "High Lifetime Value, High-Value Customers",
    2: "Engaged, Cross-Sell Potential Customers",
    3: "Moderate Lifetime Value, Engaged Customers"
}

# Replace the numeric cluster labels with descriptive labels
df['Cluster'] = df['Cluster'].replace(cluster_labels)

# Display the first few rows to check the updated labels
df.head()
df.info()


# Save the final DataFrame to a CSV file
df.to_csv('pca_cluster70_with_name.csv', encoding='utf-8', index=False)















"""

Characteristics Of Cluster : 


Cluster 0: "Low Lifetime Value, Basic Customers"
Avg Lifetime Value: Lowest across clusters (mean: $4,421).
Total Policies: Only around 1.19 policies per customer, indicating minimal engagement.
Avg Monthly Premium: Relatively low (mean: $87.29), suggesting basic policies.
Avg Income: Moderate income (mean: $36,633).
High-Value Customer: 0% high-value customers.
Cross-Sell Potential: None.
Retention Risk: Low risk.
Characteristics: This cluster represents customers with low engagement, low premiums, and no cross-sell potential. 
These customers have basic policies and are not considered high-value.








Cluster 1: "High Lifetime Value, High-Value Customers"
Avg Lifetime Value: Highest across all clusters (mean: $14,669).
Total Policies: Low to moderate (mean: 1.99 policies).
Avg Monthly Premium: Highest premium (mean: $102.92), indicating these customers are opting for higher-tier policies.
Avg Income: Slightly higher income than Cluster 0 (mean: $39,124).
High-Value Customer: 100% of the customers are high-value.
Cross-Sell Potential: None, which might indicate they already have what they need.
Retention Risk: Low risk.
Characteristics: This cluster represents high-value customers with significant lifetime value and premium policies. 
Although they have fewer policies, they tend to opt for higher premiums.






Cluster 2: "Engaged, Cross-Sell Potential Customers"
Avg Lifetime Value: Moderate (mean: $7,070).
Total Policies: High (mean: 5.28 policies), indicating strong engagement.
Avg Monthly Premium: Moderate premium (mean: $91.21).
Avg Income: Moderate (mean: $37,785).
High-Value Customer: Only about 9% are high-value.
Cross-Sell Potential: 100% cross-sell potential, making them prime candidates for upselling.
Retention Risk: Low risk.
Potential Revenue Increase: $91.64 (significant potential for revenue growth).
Characteristics: These customers are highly engaged with multiple policies, moderate premiums, and substantial cross-sell potential. 
They represent a great opportunity for increasing revenue through additional products.







Cluster 3: "Moderate Lifetime Value, Engaged Customers"
Avg Lifetime Value: Moderate (mean: $7,184).
Total Policies: Moderate (mean: 2.82 policies).
Avg Monthly Premium: Moderate (mean: $91.73).
Avg Income: Slightly higher income than Cluster 2 (mean: $38,544).
High-Value Customer: Only about 23% are high-value.
Cross-Sell Potential: About 34% have cross-sell potential.
Retention Risk: Low risk.
Potential Revenue Increase: $30.22.
Characteristics: These customers are moderately engaged with multiple policies and some cross-sell potential. They have moderate premiums and 
income, making them stable and engaged but with lower revenue growth potential compared to Cluster 2.




"""














"""


Insights:
Cluster 0 has the lowest value customers with minimal cross-sell potential. These are basic customers who could be targeted for retention strategies.



Cluster 1 represents the most valuable segment, with customers who have high premiums and are high-value but don't have cross-sell potential. 
These customers need to be nurtured for long-term retention.



Cluster 2 offers the greatest opportunity for cross-selling and revenue growth, as these customers are engaged with multiple policies.



Cluster 3 consists of stable, moderately engaged customers with moderate cross-sell potential. While less valuable than Cluster 2, 
they represent a reliable customer base


"""













"""


Benefits and Impact of the Solution


1. Improved Customer Understanding
By clustering customers based on their behavior, premiums, income, and value, the business gains a more detailed understanding of customer segments. 
This enables better decision-making and allows for more personalized and effective strategies for each customer group.

Impact: Tailored marketing and customer engagement strategies can be created, improving customer satisfaction and increasing loyalty.





2. Enhanced Cross-Selling Opportunities
The solution identifies customers with strong cross-sell potential, especially in Cluster 2. These customers already have multiple policies and 
are primed for additional products.

Impact: By targeting these customers with relevant offers, the business can maximize revenue from existing customers, improving profitability 
with minimal acquisition costs.





3. Optimized Resource Allocation
Knowing which clusters are high-value customers (e.g., Cluster 1) allows the business to focus its resources—whether in terms of customer service, 
retention strategies, or premium offerings—on segments that provide the most long-term value.

Impact: The business can optimize its marketing spend, improving cost-efficiency by focusing on high-value customers and high cross-sell potential 
segments.






4. Retention Strategy Development
The low retention risk across all clusters is a positive signal, but targeted strategies can further improve retention, particularly for 
high-value customers in Cluster 1. By understanding these segments' needs, the business can prevent churn.

Impact: Increased customer retention leads to long-term profitability. It's far cheaper to retain existing customers than to acquire new ones, 
and with the retention risk identified, the business can act proactively to reduce churn.






5. Revenue Growth
The solution highlights segments where Potential Revenue Increase is possible. For instance, Cluster 2 shows significant room for 
increasing revenues via cross-selling. This insight allows the business to prioritize campaigns that directly impact their bottom line.

Impact: Strategic upselling/cross-selling initiatives can boost revenue streams, meeting the company's financial goals.






6. Customized Product Offerings
Understanding the preferences and needs of different clusters enables the business to customize its insurance products. 
For example, customers in Cluster 1 may benefit from premium offerings, while customers in Cluster 0 might be open to more basic or budget-friendly policies.

Impact: Product customization increases the likelihood of conversion, enhancing the customer's 
experience and making the business more competitive in the market.






7. Long-Term Customer Relationship Building
The segmentation allows the business to foster long-term relationships with key customer segments, particularly those identified as 
high-value or with cross-sell potential. This builds loyalty and can lead to referral business, enhancing growth.

Impact: Better relationship-building leads to brand advocacy and long-term revenue stability through loyal, high-value customers.





8. Data-Driven Decision Making
This solution provides the client with actionable insights based on data, making their business decisions more informed. Whether 
it's adjusting premium prices, launching marketing campaigns, or creating new products, these decisions are backed by solid data.

Impact: Data-driven strategies reduce risk and improve the success rate of new initiatives, leading to more efficient operations and higher profitability.







"""





