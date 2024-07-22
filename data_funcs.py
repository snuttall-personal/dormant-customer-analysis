import pandas as pd #type hints ?
from sklearn.preprocessing import MinMaxScaler


def preprocess(customer_ref, data_ref): 
    """_summary_ ... to do insert docstrings etc

    Args:
        customer_ref (_type_): _description_
        data_ref (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Load customer data
    customer_data = pd.read_csv(customer_ref)
    # Load order data
    order_data = pd.read_csv(data_ref)

    # checked nulls for customer data, we had 6 nulls which were removed.
    # print(customer_data.isnull().sum())
    customer_data = customer_data.dropna(subset=['FIRST_MOBILE_APP_OS'])

    # checked nulls for order data and need to remove records without merchant
    # print(order_data.isnull().sum())
    order_data = order_data.dropna(subset=['MERCHANT_CATEGORY'])

    # check for duplicates - there are no duplicates in customer data
    # print(customer_data.duplicated().sum())
    
    # we have duplicates in order data
    duplicated_order_ids = order_data[order_data.duplicated('ORDER_ID', keep=False)]

    # Group by ORDER_ID and check for consistency in ACCOUNT_ID, ORDER_TIMESTAMP, ORDER_METHOD, and MERCHANT_CATEGORY
    # If so we just take first value of these fields and average the order amount
    grouped_duplicates = duplicated_order_ids.groupby('ORDER_ID').agg({
        'ACCOUNT_ID': 'nunique',
        'ORDER_TIMESTAMP': 'nunique',
        'ORDER_METHOD': 'nunique',
        'MERCHANT_CATEGORY': 'nunique'
    }).reset_index()

    # Filter out the consistent rows (where each field has only one unique value)
    inconsistent_orders = grouped_duplicates[(grouped_duplicates['ACCOUNT_ID'] > 1) |
                                            (grouped_duplicates['ORDER_TIMESTAMP'] > 1) |
                                            (grouped_duplicates['ORDER_METHOD'] > 1) |
                                            (grouped_duplicates['MERCHANT_CATEGORY'] > 1)]
    # Display inconsistent orders
    # print("Inconsistent Orders:")
    # print(inconsistent_orders) # every duplicated order ID has a single mapping to Account, timestamp, order method, merchant category
    # so we can just average the order amounts

    # Identify duplicated ORDER_ID
    duplicated_order_ids = order_data[order_data.duplicated('ORDER_ID', keep=False)]

    # Group by ORDER_ID and calculate the AVERAGE the ORDER_AMOUNT, keeping the first value for other columns
    order_data_avg = duplicated_order_ids.groupby('ORDER_ID').agg({
        'ACCOUNT_ID': 'first',
        'ORDER_TIMESTAMP': 'first',
        'ORDER_METHOD': 'first',
        'MERCHANT_CATEGORY': 'first',
        'ORDER_AMOUNT': 'mean'
    }).reset_index()

    # drop original duplicates and append the averaged duplicates which are now not duplicated any more
    order_data_no_duplicates = order_data.drop_duplicates('ORDER_ID', keep=False)
    order_data_cleaned = pd.concat([order_data_no_duplicates, order_data_avg], ignore_index=True)


    # Convert data types
    order_data_cleaned['ORDER_TIMESTAMP'] = pd.to_datetime(order_data_cleaned['ORDER_TIMESTAMP'])
    order_data_cleaned['ACCOUNT_ID'] = order_data_cleaned['ACCOUNT_ID'].astype(str)
    customer_data['ACCOUNT_ID'] = customer_data['ACCOUNT_ID'].astype(str)

    return customer_data, order_data_cleaned





def cluster_feature_preprocess(order_data_merged):
    """
    1. Assign recency of transaction for each user's transaction history
    2. Calculate frequency of transaction within each category per user
    3. Convert recency to a weighting value
    4. Combine weighted recency and frequency into preference_score
    5. Normalise score per user

    Args:
        order_data_merged (_type_): _description_

    Returns:
        _type_: _description_
    """


    #1. Calc recency
    # Calculate the most recent purchase date for each user
    most_recent_purchase = order_data_merged.groupby('ACCOUNT_ID')['ORDER_TIMESTAMP'].max().reset_index(name='most_recent_purchase')

    # Merge the most recent purchase date with the original DataFrame
    order_data_merged = pd.merge(order_data_merged, most_recent_purchase, on='ACCOUNT_ID')

    # Calculate the recency relative to the user's most recent purchase
    order_data_merged['recency'] = (order_data_merged['most_recent_purchase'] - order_data_merged['ORDER_TIMESTAMP']).dt.days

    # Calculate the average recency for each user-category combination
    recency_df = order_data_merged.groupby(['ACCOUNT_ID', 'New Category'])['recency'].mean().reset_index(name='average_recency')


    #2. calc frequency
    # Calculate the frequency of purchases for each user-category combination
    frequency_df = order_data_merged.groupby(['ACCOUNT_ID', 'New Category']).size().reset_index(name='frequency')

    # Merge the frequency and recency dataframes
    user_category_df = pd.merge(frequency_df, recency_df, on=['ACCOUNT_ID', 'New Category'])


    #3. convert recency into a weighted value
    # Function to scale within each group using MinMaxScaler
    def scale_group(group):
        scaler = MinMaxScaler()
        group[['average_recency']] = scaler.fit_transform(group[['average_recency']])
        group['weighted_recency'] = (1 - group['average_recency']).round(6)  # Higher value means more recent
        return group

    # Normalize the recency scores (higher recency means lower days) for each user
    user_category_df = user_category_df.groupby('ACCOUNT_ID').apply(scale_group).reset_index(drop=True)


    #4. combine into preference score and normalise
    # Combine frequency and weighted recency to get the preference score
    user_category_df['preference_score'] = user_category_df['frequency'] * user_category_df['weighted_recency']

    # Normalize the preference score for each user so they sum to 1
    user_category_df['preference_score'] = user_category_df.groupby('ACCOUNT_ID')['preference_score'].transform(lambda x: x / x.sum())

    # Round the preference score to avoid scientific notation
    user_category_df['preference_score'] = user_category_df['preference_score'].round(6)


    return user_category_df