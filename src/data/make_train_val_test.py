import pandas as pd
import sys
sys.path.append('D:/Innowise/DS project')
import config
from sklearn.preprocessing import LabelEncoder
from src.features.build_features import FeatureEngineering
from src.utils import get_data, downcast

prepared_data = pd.read_csv(config.prepared_data_path)
clip_threshold = config.clip_threshold
test_block_num = config.test_block_num

label_encoder = LabelEncoder()

# Пустые списки для хранения данных
X_train_list = []
y_train_list = []

# In the cycle we go through a window of 12 months
# with a step of 3 month and form parts of the dataset for subsequent cocatenation
for end_block_num in range(test_block_num, -1, -3):
    if end_block_num - 12 >= 0:
        start_block_num = end_block_num - 12
        processor = FeatureEngineering(start_block_num=start_block_num,
                                    end_block_num=end_block_num,
                                    label_encoder=label_encoder,
                                    clip_threshold=clip_threshold)
        train_data = processor.get_features(prepared_data)
        X_train, y_train = get_data(train_data)
        X_train_list.append(downcast(X_train))
        y_train_list.append(y_train)

# We create a test dataset where end_block_num=test_block_num and  start_block_num=end_block_num - 12
X_test = X_train_list[0]
y_test = y_train_list[0]

# We create a test dataset where end_block_num=(test_block_num - 1)  and  start_block_num=(end_block_num - 12 -1)
X_val = pd.concat([X_train_list[1], X_train_list[len(X_train_list)-1]], ignore_index=True)
y_val = pd.concat([y_train_list[1], y_train_list[len(y_train_list)-1]], ignore_index=True)

# Form a training dataset by concatenating all subsequent parts of the dataset
X_train = X_train_list[2]
y_train = y_train_list[2]

for i in range(3, len(X_train_list)):
    X_train = pd.concat([X_train, X_train_list[i]], ignore_index=True)
    y_train = pd.concat([y_train, y_train_list[i]], ignore_index=True)

# Save all the  datasets to the specified path
X_train.to_csv(config.X_train_path , index=False)
y_train.to_csv(config.y_train_path, index=False)
X_val.to_csv(config.X_val_path, index=False)
y_val.to_csv(config.y_val_path, index=False)
X_test.to_csv(config.X_test_path, index=False)
y_test.to_csv(config.y_test_path, index=False)
