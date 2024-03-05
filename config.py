item_cat_path = './data/raw/item_categories.csv'
items_path = './data/raw/items.csv'
shops_path = './data/raw/shops.csv'
sales_train_path = './data/raw/sales_train.csv'
test_path = './data/raw/test.csv'
prepared_data_path = './data/processed/prepared_data.csv'

# Dictionary of system duplicate store ids that need to be replaced
replace_dict = {0: 57, 1: 58, 11: 10, 40: 39}

# Number of month of test observation
test_block_num = 34

# Parameter for estimating the daily outlier, if exceeded by a specified number of times,it will be deleted
max_time_price = 10
max_time_cnt = 100
