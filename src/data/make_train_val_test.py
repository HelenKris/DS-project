import pandas as pd
import sys
sys.path.append('D:/Innowise/DS project')
import config
from sklearn.preprocessing import LabelEncoder
from src.features.build_features import FeatureEngineering

prepared_data = pd.read_csv(config.prepared_data_path)

def get_data(data):
    """Extracts features and labels from the given dataset.

    Args:
        data (DataFrame): The input dataset containing features and labels.

    Returns:
        tuple: A tuple containing features (X_train) and labels (y_train).
    """
    y_col_name = max([col for col in data.columns if isinstance(col, int)])
    X_train = data.drop([y_col_name], axis=1)
    y_train = data[y_col_name]
    rename_dict = {col: f'lag {i+1}' for i, col in enumerate([col for col in X_train .columns if isinstance(col, int)][::-1])}
    X_train.rename(columns=rename_dict, inplace=True)
    y_train.columns = ['y']
    return X_train, y_train

label_encoder = LabelEncoder()
clip_threshold = 20

# Create a list of ranges start_block_num and end_block_num
ranges = [(0, 12), (1, 13), (2, 14), (3, 15), (4, 16), (5, 17),
                (6, 18), (7, 19),(8, 20), (9, 21), (10, 22), (11, 23),
                (12, 24), (13, 25), (14, 26), (15, 27), (16, 28), (17, 29),
                (18, 30), (19, 31), (20, 32), (21, 33), (22, 34)]

# Пустые списки для хранения данных
X_train_list = []
y_train_list = []

# Итерация по диапазонам и выполнение FeatureEngineering и get_data
for start_block_num, end_block_num in ranges:
    processor = FeatureEngineering(start_block_num=start_block_num,
                                   end_block_num=end_block_num,
                                   label_encoder=label_encoder,
                                   clip_threshold = clip_threshold)
    train_data = processor.get_features(prepared_data)
    X_train, y_train = get_data(train_data)
    X_train_list.append(X_train)
    y_train_list.append(y_train)

X_val = pd.concat([X_train_list[0], X_train_list[5],X_train_list[15],X_train_list[20]], ignore_index=True)
y_val = pd.concat([y_train_list[0], y_train_list[5], y_train_list[15], y_train_list[20]], ignore_index=True)

X_test = pd.concat([X_train_list[22]], ignore_index=True)
y_test = pd.concat([y_train_list[22]], ignore_index=True)

X_train = pd.concat([X_train_list[1], X_train_list[2],X_train_list[3],X_train_list[4],X_train_list[6],
                     X_train_list[7],X_train_list[8],X_train_list[9],X_train_list[10], X_train_list[11],
                     X_train_list[12],X_train_list[13],X_train_list[14], X_train_list[16],X_train_list[17],
                     X_train_list[18],X_train_list[19], X_train_list[21]], ignore_index=True)
y_train = pd.concat([y_train_list[1], y_train_list[2],y_train_list[3],y_train_list[4],y_train_list[6],
                     y_train_list[7],y_train_list[8],y_train_list[9],y_train_list[10], y_train_list[11],
                     y_train_list[12],y_train_list[13],y_train_list[14], y_train_list[16],y_train_list[17],
                     y_train_list[18],y_train_list[19], y_train_list[21]], ignore_index=True)

def downcast(df, verbose=True):
    """Downcasts the data types of DataFrame columns to reduce memory usage.

    Args:
        df (DataFrame): The DataFrame to downcast.
        verbose (bool, optional): If True, prints the percentage of memory compressed. Defaults to True.

    Returns:
        DataFrame: The DataFrame with downcasted data types.
    """
    if not isinstance(df, pd.DataFrame):
        return df  # Skip if not a DataFrame
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        dtype_name = df[col].dtype.name
        if dtype_name == 'object':
            pass
        elif dtype_name == 'bool':
            df[col] = df[col].astype('int8')
        elif dtype_name.startswith('int') or (df[col].round() == df[col]).all():
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('{:.1f}% compressed'.format(100 * (start_mem - end_mem) / start_mem))
    return df

all_df = [X_val, y_val, X_test, y_test,X_train, y_train]
for df in all_df:
    if isinstance(df, pd.DataFrame):
        df = downcast(df)

X_train.to_csv(config.X_train_path , index=False)
y_train.to_csv(config.y_train_path, index=False)
X_val.to_csv(config.X_val_path, index=False)
y_val.to_csv(config.y_val_path, index=False)
X_test.to_csv(config.X_test_path, index=False)
y_test.to_csv(config.y_test_path, index=False)
