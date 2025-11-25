import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json
from typing import Dict, List, Any, Tuple
import logging

from config import Config

class IntentClassifier:
    """
    Clasificador de intenciones usando Machine Learning
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.config = Config
        
        # Modelos
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        
        # Datos
        self.training_data = None
        self.intent_mapping = {}
        
        self.logger.info("🤖 Clasificador de intenciones inicializado")
    
    def _setup_logger(self):
        logger = logging.getLogger('IntentClassifier')
        logger.setLevel(logging.INFO)
        return logger
    
    def prepare_training_data(self, intents_data: List[Dict]) -> pd.DataFrame:
        """
        Preparar datos de entrenamiento a partir de intents JSON
        """
        training_examples = []
        
        for intent in intents_data:
            tag = intent["tag"]
            patterns = intent["patterns"]
            
            for pattern in patterns:
                training_examples.append({
                    "text": pattern,
                    "tag": tag
                })
        
        df = pd.DataFrame(training_examples)
        self.logger.info(f"📊 Datos preparados: {len(df)} ejemplos, {len(df['tag'].unique())} clases")
        
        return df
    
    def train_model(self, intents_data: List[Dict], test_size: float = 0.2) -> Dict[str, Any]:
        """
        Entrenar modelo de clasificación de intenciones
        """
        try:
            # Preparar datos
            self.training_data = self.prepare_training_data(intents_data)
            
            if self.training_data.empty:
                raise ValueError("No hay datos de entrenamiento")
            
            # Dividir datos
            X = self.training_data["text"]
            y = self.training_data["tag"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.config.ML["RANDOM_STATE"], stratify=y
            )
            
            # Vectorizar textos
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.ML["TFIDF_MAX_FEATURES"],
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Seleccionar y entrenar modelo
            classifier_type = self.config.ML["CLASSIFIER_TYPE"]
            
            if classifier_type == "svm":
                self.classifier = SVC(kernel='linear', probability=True, random_state=42)
            elif classifier_type == "random_forest":
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            elif classifier_type == "logistic_regression":
                self.classifier = LogisticRegression(random_state=42, max_iter=1000)
            else:
                self.classifier = SVC(kernel='linear', probability=True, random_state=42)
            
            self.classifier.fit(X_train_vec, y_train)
            
            # Evaluar modelo
            y_pred = self.classifier.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Crear mapeo de intents
            self.intent_mapping = {tag: idx for idx, tag in enumerate(self.classifier.classes_)}
            
            self.is_trained = True
            
            results = {
                "accuracy": accuracy,
                "training_size": len(X_train),
                "test_size": len(X_test),
                "num_classes": len(self.classifier.classes_),
                "classifier_type": classifier_type,
                "feature_count": X_train_vec.shape[1]
            }
            
            self.logger.info(f"✅ Modelo entrenado - Accuracy: {accuracy:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error entrenando modelo: {e}")
            return {"error": str(e)}
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Predecir intent para un texto dado
        """
        if not self.is_trained or not self.vectorizer or not self.classifier:
            raise ValueError("Modelo no entrenado")
        
        # Vectorizar texto
        text_vec = self.vectorizer.transform([text])
        
        # Predecir probabilidades
        probabilities = self.classifier.predict_proba(text_vec)[0]
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_tag = self.classifier.classes_[predicted_idx]
        
        return predicted_tag, confidence
    
    def save_model(self, model_dir: str = None):
        """Guardar modelo entrenado"""
        if not model_dir:
            model_dir = self.config.MODELS_DIR
        
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Guardar vectorizador
            vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
            joblib.dump(self.vectorizer, vectorizer_path)
            
            # Guardar clasificador
            classifier_path = os.path.join(model_dir, 'intent_classifier.pkl')
            joblib.dump(self.classifier, classifier_path)
            
            # Guardar mapeo
            mapping_path = os.path.join(model_dir, 'intent_mapping.json')
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.intent_mapping, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 Modelos guardados en: {model_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando modelos: {e}")
    
    def load_model(self, model_dir: str = None):
        """Cargar modelo entrenado"""
        if not model_dir:
            model_dir = self.config.MODELS_DIR
        
        try:
            # Cargar vectorizador
            vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
            
            # Cargar clasificador
            classifier_path = os.path.join(model_dir, 'intent_classifier.pkl')
            if os.path.exists(classifier_path):
                self.classifier = joblib.load(classifier_path)
            
            # Cargar mapeo
            mapping_path = os.path.join(model_dir, 'intent_mapping.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.intent_mapping = json.load(f)
            
            self.is_trained = True
            self.logger.info("✅ Modelos cargados correctamente")
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando modelos: {e}")
            self.is_trained = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "classifier_type": type(self.classifier).__name__,
            "num_classes": len(self.classifier.classes_),
            "feature_count": self.vectorizer.get_feature_names_out().shape[0],
            "intent_mapping": self.intent_mapping
        }

# Instancia global del clasificador
intent_classifier = IntentClassifier()