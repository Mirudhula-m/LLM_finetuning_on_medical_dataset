from sagemaker_setup import setup_sagemaker

class CFG:

    project = "LLM finetuning"
    seed = 42

    # DATA RELATED CONFIG

    dataset_name = "BI55/MedText"
    val_split = 0.1
    
    # AWS SAGEMAKER RELATED CONFIG

    sess = setup_sagemaker()
    s3_save_path = f"s3://{sess.default_bucket()}/datasets/llama3"