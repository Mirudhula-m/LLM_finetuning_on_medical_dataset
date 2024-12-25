import pandas as pd
from datasets import load_dataset, Dataset

from config import CFG

def save_data_to_s3():

    # this data has only train set. so we will split this into train and test set
    dataset = load_dataset(CFG.dataset_name, split = 'train')
    n_data = len(dataset)
    n_data_train = int((1 - CFG.val_split) * n_data)    
    train_dataset = Dataset.from_dict(dataset[:n_data_train])
    valid_dataset = Dataset.from_dict(dataset[n_data_train:n_data])

    print('Train = ',len(train_dataset), '| Valid = ', len(valid_dataset))

    # save data to s3 buckets
    input_path = CFG.s3_save_path
    train_dataset_s3_path = f"{input_path}/train/dataset.json"
    train_dataset.to_json(train_dataset_s3_path, orient="records")
    valid_dataset_s3_path = f"{input_path}/val/dataset.json"
    valid_dataset.to_json(valid_dataset_s3_path, orient="records")

    print(f"Training data uploaded to:")
    print(train_dataset_s3_path)
    print(valid_dataset_s3_path)
    print(f"https://s3.console.aws.amazon.com/s3/buckets/{CFG.sess.default_bucket()}/?region={CFG.sess.boto_region_name}&prefix={input_path.split('/', 3)[-1]}/")