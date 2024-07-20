#preprocessing
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the data
df = pd.read_csv('/content/sample_data/Datasetfinal.csv')
print(df.info())


# Step 3: Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
print(df.head())
print(df.info())

# Step 4: Check for missing values
print(df.isnull().sum())

# Step 5: Visualize box plots for outlier detection
fig, axs = plt.subplots(9, 1, dpi=95, figsize=(7, 17))
i = 0
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        axs[i].boxplot(df[col], vert=False)
        axs[i].set_ylabel(col)
        i += 1
plt.show()

# Step 6: Remove outliers for specific columns (optional)
columns_to_clean = ['Views', 'Demand', 'Inventory']
for column in columns_to_clean:
    q1, q3 = np.percentile(df[column], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Step 7: Calculate and visualize the correlation matrix
corr = df.corr()
plt.figure(dpi=130)
sns.heatmap(corr, annot=True, fmt='.2f')
plt.show()

# Step 8: Extract the 'Date' column
dates = df['Date']

# Step 9: Drop the 'Date' column from the DataFrame for scaling
X = df.drop(columns=['Date'])

# Step 10: Custom scaling function
def custom_scale(df, scale_factors):
    scaled_df = df.copy()
    for col, factor in scale_factors.items():
        scaled_df[col] = (scaled_df[col] / factor).round() * factor
    return scaled_df

# Define scale factors for each column
scale_factors = {
    'Views': 1000,
    'Demand': 10,
    'Inventory': 1000
}

# Apply the custom scaling
rescaledX_df = custom_scale(X, scale_factors)

# Step 11: Reattach the 'Date' column
rescaledX_df['Date'] = dates.values
# Step 12: Split the data into training and testing sets
split_point = int(len(rescaledX_df) * 0.92)
train_df = rescaledX_df[:split_point]
test_df = rescaledX_df[split_point:]

# Display the processed and integrated DataFrame
print(rescaledX_df.head())
print(rescaledX_df.info())
print(train_df.head())
print(test_df.head())
print(train_df.shape)
print(test_df.shape)