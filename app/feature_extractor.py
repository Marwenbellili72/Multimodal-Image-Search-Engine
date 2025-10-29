from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class FeatureExtractor:
    def __init__(self):
        logging.info("Initialisation du Feature Extractor (VGG16)...")
        try:
            base_model = VGG16(weights='imagenet', include_top=True)
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
            logging.info("Modèle VGG16 chargé et tronqué à fc1.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement de VGG16 : {e}")
            self.model = None

    def extract(self, img):
        if self.model is None:
            logging.error("Le modèle VGG16 n'est pas disponible.")
            return np.array([])
            
        img = img.resize((224, 224)) 
        img = img.convert('RGB') 
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0) 
        x = preprocess_input(x)         
        feature = self.model.predict(x, verbose=0)[0]         
        norm = np.linalg.norm(feature)
        if norm == 0:
            return feature 
        return feature / norm