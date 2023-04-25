import os


def prepare():
    '''
    this function is used to prepare the environment variables
    in order to save the model to a different path
    :return:
    '''
    os.environ['HF_DATASETS_CACHE'] = '../cache/hf_cache/datasets'
    os.environ['HF_METRICS_CACHE'] = '../cache/hf_cache/metrics'
    os.environ['HF_MODULES_CACHE'] = '../cache/hf_cache/modules'
    os.environ['HF_DATASETS_DOWNLOADED_EVALUATE_PATH'] = '../cache/hf_cache/datasets_downloaded_evaluate'
    os.environ['TRANSFORMERS_CACHE'] = '../cache/transformers_cache'
    os.environ['TORCH_HOME'] = '../cache/torch_home'