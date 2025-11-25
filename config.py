import os
from typing import Dict, Any

class Config:
    """Configuración avanzada con soporte NLP"""
    
    # Configuración de la aplicación
    DEBUG = True
    SECRET_KEY = 'chatbot-nlp-secret-key-2024'
    
    # Rutas de archivos
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INTENTS_FILE = os.path.join(BASE_DIR, 'data', 'intents.json')
    TRAINING_DATA_FILE = os.path.join(BASE_DIR, 'data', 'training_data', 'labeled_intents.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    CONVERSATIONS_DIR = os.path.join(BASE_DIR, 'data', 'conversations')
    
    # Configuración del chatbot
    MIN_CONFIDENCE = 0.6  # Más estricto con NLP
    ENABLE_LOGGING = True
    
    # ✅ CONFIGURACIÓN NLP AVANZADA
    NLP = {
        "USE_ADVANCED_NLP": True,
        "MODEL_TYPE": "hybrid",  # hybrid, ml, semantic
        "SPACY_MODEL": "es_core_news_sm",
        "SIMILARITY_THRESHOLD": 0.75,
        "ENABLE_ENTITY_EXTRACTION": True,
        "ENABLE_SENTIMENT_ANALYSIS": True,
        "USE_WORD_EMBEDDINGS": True,
    }
    
    # Configuración Machine Learning
    ML = {
        "CLASSIFIER_TYPE": "svm",  # svm, random_forest, logistic_regression
        "TFIDF_MAX_FEATURES": 2000,
        "TEST_SIZE": 0.2,
        "RANDOM_STATE": 42,
    }
    
    # Configuración de características NLP
    FEATURES = {
        "USE_LEMMATIZATION": True,
        "REMOVE_STOPWORDS": True,
        "USE_STEMMING": False,
        "EXTRACT_ENTITIES": True,
        "ANALYZE_SENTIMENT": True,
        "CALCULATE_READABILITY": True,
    }

    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        return {
            key: value for key, value in cls.__dict__.items() 
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def ensure_directories(cls):
        """Crear directorios necesarios"""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.TRAINING_DATA_FILE), exist_ok=True)
        os.makedirs(cls.CONVERSATIONS_DIR, exist_ok=True)