from config import CFG

def save_data_to_s3(train_dataset, valid_dataset):
    # save data to s3 buckets
    input_path = CFG.s3_save_path
    train_dataset_s3_path = f"{input_path}/train/dataset.json"
    train_dataset.to_json(train_dataset_s3_path, orient="records")
    valid_dataset_s3_path = f"{input_path}/val/dataset.json"
    valid_dataset.to_json(valid_dataset_s3_path, orient="records")

    CFG.train_dataset_s3_path = train_dataset_s3_path
    CFG.valid_dataset_s3_path = valid_dataset_s3_path

    print(f"Training data uploaded to:")
    print(train_dataset_s3_path)
    print(valid_dataset_s3_path)
    print(f"https://s3.console.aws.amazon.com/s3/buckets/{CFG.sess.default_bucket()}/?region={CFG.sess.boto_region_name}&prefix={input_path.split('/', 3)[-1]}/")