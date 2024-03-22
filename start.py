import config
from sklearn.preprocessing import LabelEncoder

from src.data.prepare_datasets import prepare_datasets
from src.data.get_dataset import ELT

item_cat_path = config.item_cat_path
items_path = config.items_path
shops_path = config.shops_path
sales_train_path = config.sales_train_path
test_path = config.test_path
prepared_data_path = config.prepared_data_path
replace_dict = config.replace_dict
max_time_cnt = config.max_time_cnt
max_time_price = config.max_time_price
clip_threshold = config.clip_threshold
output_filepaths = [
    config.X_train_path,
    config.y_train_path,
    config.X_val_path,
    config.y_val_path,
    config.X_test_path,
]
label_encoder = LabelEncoder()


if __name__ == "__main__":
    ELT(
        item_cat_path=config.item_cat_path,
        items_path=config.items_path,
        shops_path=config.shops_path,
        sales_train_path=config.sales_train_path,
        test_path=config.test_path,
        prepared_data_path=config.prepared_data_path,
        replace_dict=config.replace_dict,
        max_time_cnt=config.max_time_cnt,
        max_time_price=config.max_time_price,
    ).transform()
    prepare_datasets(
        input_filepath=prepared_data_path,
        output_filepaths=output_filepaths,
        label_encoder=label_encoder,
        clip_threshold=clip_threshold,
    )
