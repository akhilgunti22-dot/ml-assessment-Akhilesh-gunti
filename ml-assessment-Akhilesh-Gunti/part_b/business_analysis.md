Part B: Business Case Analysis
B1. Problem Formulation
(a) Machine Learning Formulation

The objective is to predict items_sold, which serves as the target variable.

The input features include:

Store attributes: store_size, location_type, competition_density
Promotion details: promotion_type
Temporal features: month, year, day_of_week, is_weekend, is_festival
Historical behavior: past sales (if available)

This is a supervised learning regression problem, as the goal is to predict a continuous numerical value (items sold). Regression is appropriate because the output is not categorical but represents quantity, and the business objective is to maximize sales volume.

(b) Why Items Sold is Better than Revenue

Using items sold (sales volume) is more reliable than revenue because revenue can be heavily influenced by pricing changes, discounts, or product mix. For example, a high discount promotion may increase units sold but reduce total revenue, making revenue an inconsistent performance metric.

This illustrates the broader principle that the target variable should directly reflect the business objective. In this case, the goal is to evaluate promotion effectiveness in driving customer purchases, which is better captured by volume rather than monetary value.

(c) Alternative Modelling Strategy

Instead of using a single global model, a better approach is to use a segmented or hierarchical modelling strategy. Stores can be grouped based on location type (urban, semi-urban, rural) or customer behavior, and separate models can be trained for each segment.

This approach accounts for heterogeneity across stores, as different customer demographics and competition levels lead to different responses to promotions. It improves model accuracy and produces more relevant recommendations.

B2. Data and EDA Strategy
(a) Data Joining and Dataset Design

The four tables (transactions, store attributes, promotion details, and calendar) would be joined using common keys such as:

store_id (to link store attributes)
promotion_id or promotion_type
transaction_date (to join with calendar)

The final dataset should have a monthly store-level grain:
One row = one store in one month

Before modelling, the following aggregations would be performed:

Total items_sold per store per month
Average basket size
Total transactions / footfall
Promotion applied during that month
Calendar features (weekends, festivals)

This aggregation ensures consistency between features and the prediction target.

(b) Exploratory Data Analysis (EDA)
Sales by Promotion Type (Bar Chart)
Helps identify which promotions drive higher sales.
→ Guides feature importance and promotion effectiveness.
Sales Trends Over Time (Line Plot)
Reveals seasonality and trends across months or years.
→ Helps create time-based features like month or season.
Distribution of Items Sold (Histogram)
Shows skewness or outliers in the target variable.
→ May require transformation or outlier handling.
Correlation Heatmap
Identifies relationships between numerical variables.
→ Helps in feature selection and reducing redundancy.
Sales by Store Type (Box Plot)
Compares performance across urban, semi-urban, and rural stores.
→ Supports segmentation strategy.
(c) Handling Promotion Imbalance

Since 80% of transactions occur without promotions, the model may become biased toward predicting outcomes for non-promotion cases.

To address this:

Use resampling techniques (oversample promotion cases or undersample non-promotion cases)
Apply feature weighting or stratified analysis
Evaluate model performance separately for promotion vs non-promotion scenarios

This ensures the model learns meaningful patterns for promotion effectiveness.

B3. Model Evaluation and Deployment
(a) Train-Test Split and Metrics

The dataset should be split using a time-based split, where earlier data is used for training and the most recent months are used for testing.

A random split is inappropriate because it causes data leakage, allowing future information to influence past predictions.

Evaluation metrics:

RMSE: Measures large errors more heavily; useful for identifying major prediction mistakes
MAE: Provides average error magnitude; easier to interpret in business terms

Both metrics help assess how accurately the model predicts sales volume.

(b) Explaining Model Recommendations

Feature importance can be used to understand why different promotions are recommended for the same store in different months.

For example:

In December, features like festival season and high customer demand may increase the effectiveness of Loyalty Points Bonus
In March, lower demand or different customer behavior may favor Flat Discount

By analyzing feature contributions (e.g., using feature importance or SHAP values), we can explain how seasonality, promotions, and store context interact to influence recommendations. This helps build trust with the marketing team.

(c) Deployment and Monitoring

Deployment Process:

Train the model and save it using a serialization method (e.g., pickle or joblib)
At the start of each month, collect new data (store info, calendar features, promotions)
Apply the same preprocessing pipeline
Generate predictions for all stores

Monitoring:

Track prediction errors over time (RMSE/MAE)
Monitor data drift (changes in feature distributions)
Compare predicted vs actual sales monthly

If performance degrades significantly, retraining should be triggered using updated data.
