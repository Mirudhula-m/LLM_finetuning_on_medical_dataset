from sagemaker_setup import setup_sagemaker
from sagemaker.s3 import S3Uploader

def model_yaml_2_S3():
    train_config_s3_path = S3Uploader.uppload(local_path=CFG.model_yaml, desired_s3_uri=f"{CFG.s3_save_path}/config")
    print(f"Training config uploaded to:", train_config_s3_path)
    return train_config_s3_path

class CFG:

    project = "LLM finetuning"
    seed = 42

    # DATA RELATED CONFIG

    dataset_name = "BI55/MedText"
    val_split = 0.1
    
    # AWS SAGEMAKER RELATED CONFIG

    sess = setup_sagemaker()

    # AWS S3 RELATED CONFIG

    s3_save_path = f"s3://{sess.default_bucket()}/datasets/llama3"
    model_yaml = "llama_3_70b_fsdp_qlora.yaml"
    train_config_s3_path = model_yaml_2_S3()
    
    
