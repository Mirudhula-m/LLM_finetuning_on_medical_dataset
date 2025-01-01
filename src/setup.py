from sagemaker_setup import setup_sagemaker
from sagemaker.s3 import S3Uploader
from sagemaker.huggingface import HuggingFace
from huggingface_hub import HfFolder

def model_yaml_2_S3(model_yaml, s3_save_path):
    train_config_s3_path = S3Uploader.upload(local_path=model_yaml, desired_s3_uri=f"{s3_save_path}/config")
    print(f"Training config uploaded to:", train_config_s3_path)
    return train_config_s3_path

class CFG:

    project = "LLM finetuning"
    seed = 42

    # DATA RELATED CONFIG

    dataset_name = "BI55/MedText"
    val_split = 0.1
    
    # AWS SAGEMAKER RELATED CONFIG

    sess, role = setup_sagemaker()

    # AWS S3 RELATED CONFIG

    s3_save_path = f"s3://{sess.default_bucket()}/datasets/llama3"
    model_yaml = "llama_3_70b_fsdp_qlora.yaml"
    train_config_s3_path = model_yaml_2_S3(model_yaml, s3_save_path)

    # TRAINING JOB NAME
    job_name = f'llama3-70b-exp1'
    
    # HUGGING FACE ESTIMATOR CONFIG

    huggingface_estimator = HuggingFace(
        entry_point          = 'run_fsdp_qlora.py',      # train script
        source_dir           = '../scripts',  # directory which includes all the files needed for training
        instance_type        = 'ml.p4d.24xlarge',  # instances type used for the training job
        instance_count       = 1,                 # the number of instances used for training
        max_run              = 2*24*60*60,        # maximum runtime in seconds (days * hours * minutes * seconds)
        base_job_name        = job_name,          # the name of the training job
        role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
        volume_size          = 500,               # the size of the EBS volume in GB
        transformers_version = '4.36.0',          # the transformers version used in the training job
        pytorch_version      = '2.1.0',           # the pytorch_version version used in the training job
        py_version           = 'py310',           # the python version used in the training job
        hyperparameters      =  {
            "config": "/opt/ml/input/data/config/llama_3_70b_fsdp_qlora.yaml" # path to TRL config which was uploaded to s3
        },
        disable_output_compression = True,        # not compress output to save training time and cost
        distribution={"torch_distributed": {"enabled": True}},   # enables torchrun
        environment  = {
            "HUGGINGFACE_HUB_CACHE": "/tmp/.cache", # set env variable to cache models in /tmp
            "HF_TOKEN": HfFolder.get_token(),       # huggingface token to access gated models, e.g. llama 3
            "ACCELERATE_USE_FSDP": "1",             # enable FSDP
            "FSDP_CPU_RAM_EFFICIENT_LOADING": "1"   # enable CPU RAM efficient loading
        }, 
    )

    # SAGEMAKER DEPLOY CONFIG

    instance_type = "ml.p4d.24xlarge"
    health_check_timeout = 1200 # 20 minutes    

    # MODEL AND ENDPOINT CONFIGURATION PARAMETER
    # Define Model and Endpoint configuration parameter
    endpoint_config = {
    'HF_MODEL_ID': "/opt/ml/model",       # Path to the model in the container
    'SM_NUM_GPUS': "8",                   # Number of GPU used per replica
    'MAX_INPUT_LENGTH': "8000",           # Max length of input text
    'MAX_TOTAL_TOKENS': "8096",           # Max length of the generation (including input text)
    'MAX_BATCH_PREFILL_TOKENS': "16182",  # Limits the number of tokens that can be processed in parallel during the generation
    'MESSAGES_API_ENABLED': "true",       # Enable the OpenAI Messages API
    }