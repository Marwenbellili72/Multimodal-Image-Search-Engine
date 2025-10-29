import os
# =========================================================
IMAGES_DIR = r"D:\images" 
INDEX_NAME = 'my-image-embeddings'
ES_HOSTS = [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}]
ES_TIMEOUT = 60
METRIC_SOURCE = "cosineSimilarity(params.query_vector, 'image_embedding') + 1.0"
METRIC_NAME = "Distance Cosinus (Similitude)"
TOP_N_DEFAULT = 5