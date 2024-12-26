from setup import CFG
from src.hugging_face_estimator import huggingface_estimator
from sagemaker.huggingface import get_huggingface_llm_image_uri
from huggingface_hub import HfFolder
from sagemaker.huggingface import HuggingFaceModel
from datasets import load_dataset, Dataset
from src.get_data import save_data_to_s3

# define a data input dictonary with our uploaded s3 uris
def llm_estimator():
  data = {
    'train': CFG.train_dataset_s3_path,
    'test': CFG.test_dataset_s3_path,
    'config': CFG.train_config_s3_path
    }

  # starting the train job with our uploaded datasets as input
  huggingface_estimator.fit(data, wait=True)
  return huggingface_estimator

def create_llm_image():
  # retrieve the llm image uri
  llm_image = get_huggingface_llm_image_uri(
    "huggingface",
    session=CFG.sess,
    version="2.0.2",
  )

  # print ecr image uri
  print(f"llm image uri: {llm_image}")

  return llm_image


def hugging_face_model():
  # create HuggingFaceModel with the image uri
  llm_model = HuggingFaceModel(
    role=CFG.role,
    # path to s3 bucket with model, we are not using a compressed model
    # {'S3DataSource':{'S3Uri': "s3://...",'S3DataType': 'S3Prefix','CompressionType': 'None'}},
    model_data=huggingface_estimator.model_data,
    image_uri=create_llm_image(),
    env=CFG.endpoint_config
  )
  return llm_model

def deploy_llm():
  # get data save it to s3 buckets
  # this data has only train set. so we will split this into train and test set
  dataset = load_dataset(CFG.dataset_name, split = 'train')
  n_data = len(dataset)
  n_data_train = int((1 - CFG.val_split) * n_data)    
  train_dataset = Dataset.from_dict(dataset[:n_data_train])
  valid_dataset = Dataset.from_dict(dataset[n_data_train:n_data])

  print('Train = ',len(train_dataset), '| Valid = ', len(valid_dataset))

  save_data_to_s3(train_dataset, valid_dataset)
  
  # Deploy model to an endpoint
  llm = hugging_face_model().deploy(
    initial_instance_count=1,
    instance_type=CFG.instance_type,
    container_startup_health_check_timeout=CFG.health_check_timeout, # 20 minutes to give SageMaker the time to download and merge model
  )

