import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

from .utils.text_processor import text_processor
from config import Config

class NLEngine:
    """
    Motor principal de Procesamiento de Lenguaje Natural
    Combina técnicas clásicas y modernas de NLP
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.config = Config
        self.text_processor = text_processor
        
        # Modelos cargados
        self.ml_classifier = None
        self.tfidf_vectorizer = None
        
        # Cargar modelos si existen
        self._load_models()
        
        self.logger.info("🚀 Motor NLP inicializado correctamente")
    
    def _setup_logger(self):
        logger = logging.getLogger('NLEngine')
        logger.setLevel(logging.INFO)
        return logger
    
    def _load_models(self):
        """Cargar modelos ML pre-entrenados"""
        try:
            classifier_path = os.path.join(self.config.MODELS_DIR, 'intent_classifier.pkl')
            vectorizer_path = os.path.join(self.config.MODELS_DIR, 'tfidf_vectorizer.pkl')
            
            if os.path.exists(classifier_path):
                self.ml_classifier = joblib.load(classifier_path)
                self.logger.info("✅ Clasificador ML cargado")
            
            if os.path.exists(vectorizer_path):
                self.tfidf_vectorizer = joblib.load(vectorizer_path)
                self.logger.info("✅ Vectorizador TF-IDF cargado")
                
        except Exception as e:
            self.logger.warning(f"❌ No se pudieron cargar modelos ML: {e}")
    
    def process_query(self, user_input: str, available_intents: List[Dict]) -> Dict[str, Any]:
        """
        Procesar consulta del usuario usando múltiples estrategias NLP
        """
        # Extraer características avanzadas
        features = self.text_processor.extract_features(user_input)
        processed_text = self.text_processor.preprocess_text(user_input, "for_ml")
        
        # Aplicar múltiples métodos de clasificación
        results = {
            "user_input": user_input,
            "processed_text": processed_text,
            "features": features,
            "classification_methods": {},
            "final_intent": None,
            "confidence": 0.0
        }
        
        # Método 1: Clasificación por ML (si está disponible)
        ml_result = self._classify_with_ml(processed_text, available_intents)
        results["classification_methods"]["machine_learning"] = ml_result
        
        # Método 2: Similitud semántica
        semantic_result = self._classify_with_semantic_similarity(user_input, available_intents)
        results["classification_methods"]["semantic_similarity"] = semantic_result
        
        # Método 3: Reglas y patrones (fallback)
        rule_result = self._classify_with_rules(user_input, available_intents, features)
        results["classification_methods"]["rule_based"] = rule_result
        
        # Combinar resultados usando ensemble
        final_result = self._ensemble_classification(results, available_intents)
        results.update(final_result)
        
        self.logger.info(f"📊 Query procesada: '{user_input}' -> {results['final_intent']} (conf: {results['confidence']:.3f})")
        
        return results
    
    def _classify_with_ml(self, processed_text: str, intents: List[Dict]) -> Dict[str, Any]:
        """Clasificación usando Machine Learning"""
        if not self.ml_classifier or not self.tfidf_vectorizer:
            return {"method": "ml", "available": False, "intent": None, "confidence": 0.0}
        
        try:
            # Vectorizar el texto
            text_vector = self.tfidf_vectorizer.transform([processed_text])
            
            # Predecir probabilidades
            probabilities = self.ml_classifier.predict_proba(text_vector)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            # Mapear a tags de intents (esto depende de cómo se entrenó el modelo)
            # Por ahora, usamos un mapeo simple
            intent_tags = [intent["tag"] for intent in intents]
            if hasattr(self.ml_classifier, 'classes_'):
                predicted_tag = self.ml_classifier.classes_[predicted_class_idx]
            else:
                predicted_tag = intent_tags[predicted_class_idx % len(intent_tags)]
            
            # Encontrar el intent correspondiente
            predicted_intent = next((intent for intent in intents if intent["tag"] == predicted_tag), None)
            
            return {
                "method": "ml",
                "available": True,
                "intent": predicted_intent,
                "confidence": confidence,
                "all_probabilities": dict(zip(intent_tags, probabilities))
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error en clasificación ML: {e}")
            return {"method": "ml", "available": False, "intent": None, "confidence": 0.0}
    
    def _classify_with_semantic_similarity(self, user_input: str, intents: List[Dict]) -> Dict[str, Any]:
        """Clasificación usando similitud semántica"""
        best_intent = None
        best_score = 0.0
        best_pattern = ""
        similarities = []
        
        for intent in intents:
            intent_scores = []
            
            for pattern in intent["patterns"]:
                similarity = self.text_processor.calculate_semantic_similarity(user_input, pattern)
                intent_scores.append(similarity)
                
                if similarity > best_score:
                    best_score = similarity
                    best_intent = intent
                    best_pattern = pattern
            
            avg_similarity = np.mean(intent_scores) if intent_scores else 0.0
            similarities.append({
                "tag": intent["tag"],
                "average_similarity": avg_similarity,
                "max_similarity": max(intent_scores) if intent_scores else 0.0
            })
        
        return {
            "method": "semantic",
            "intent": best_intent,
            "confidence": best_score,
            "matched_pattern": best_pattern,
            "all_similarities": similarities
        }
    
    def _classify_with_rules(self, user_input: str, intents: List[Dict], features: Dict) -> Dict[str, Any]:
        """Clasificación usando reglas y patrones básicos"""
        user_input_lower = user_input.lower()
        best_intent = None
        best_score = 0.0
        
        for intent in intents:
            score = 0.0
            
            # Verificar patrones simples
            for pattern in intent["patterns"]:
                pattern_lower = pattern.lower()
                
                # Coincidencia exacta
                if pattern_lower in user_input_lower:
                    score = max(score, 0.8)
                
                # Coincidencia de palabras clave
                pattern_words = set(self.text_processor.preprocess_text(pattern, "basic").split())
                user_words = set(self.text_processor.preprocess_text(user_input, "basic").split())
                
                common_words = pattern_words.intersection(user_words)
                if common_words:
                    word_score = len(common_words) / len(pattern_words) * 0.6
                    score = max(score, word_score)
            
            # Reglas contextuales
            if features["basic"]["has_greeting"] and intent["tag"] == "saludo":
                score = max(score, 0.9)
            
            if features["basic"]["has_question"] and "pregunta" in intent.get("context", ""):
                score = max(score, 0.3)
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return {
            "method": "rules",
            "intent": best_intent,
            "confidence": best_score
        }
    
    def _ensemble_classification(self, results: Dict, intents: List[Dict]) -> Dict[str, Any]:
        """Combinar resultados de múltiples métodos usando ensemble"""
        methods = results["classification_methods"]
        intent_scores = {}
        
        # Ponderaciones para cada método
        weights = {
            "machine_learning": 0.5,
            "semantic_similarity": 0.3,
            "rule_based": 0.2
        }
        
        # Acumular scores por intent
        for method_name, method_result in methods.items():
            weight = weights.get(method_name, 0.1)
            
            if method_result.get("intent"):
                intent_tag = method_result["intent"]["tag"]
                confidence = method_result["confidence"]
                
                if intent_tag not in intent_scores:
                    intent_scores[intent_tag] = 0.0
                
                intent_scores[intent_tag] += confidence * weight
        
        # Encontrar el intent con mayor score
        if intent_scores:
            best_tag = max(intent_scores, key=intent_scores.get)
            best_score = intent_scores[best_tag]
            best_intent = next((intent for intent in intents if intent["tag"] == best_tag), None)
        else:
            best_intent = None
            best_score = 0.0
        
        # Método ganador
        winning_method = "unknown"
        winning_confidence = 0.0
        
        for method_name, method_result in methods.items():
            if method_result.get("confidence", 0) > winning_confidence:
                winning_confidence = method_result["confidence"]
                winning_method = method_name
        
        return {
            "final_intent": best_intent,
            "confidence": best_score,
            "winning_method": winning_method,
            "ensemble_scores": intent_scores
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas del texto"""
        if not self.text_processor.nlp:
            return []
        
        try:
            doc = self.text_processor.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": self._get_entity_description(ent.label_)
                })
            
            return entities
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo entidades: {e}")
            return []
    
    def _get_entity_description(self, label: str) -> str:
        """Obtener descripción en español de las etiquetas de entidades"""
        entity_descriptions = {
            "PER": "Persona",
            "LOC": "Lugar",
            "ORG": "Organización",
            "MISC": "Misceláneo",
            "DATE": "Fecha",
            "TIME": "Tiempo"
        }
        return entity_descriptions.get(label, label)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Análisis de sentimiento del texto"""
        # Palabras positivas y negativas en español
        positive_words = {
            'bueno', 'excelente', 'genial', 'perfecto', 'ayuda', 'gracias', 'bien',
            'fantástico', 'maravilloso', 'útil', 'agradecido', 'agradecida', 'encantado',
            'feliz', 'contento', 'satisfecho', 'brillante', 'increíble'
        }
        
        negative_words = {
            'malo', 'problema', 'error', 'difícil', 'complicado', 'no', 'nunca',
            'terrible', 'horrible', 'pésimo', 'frustrado', 'enojado', 'molesto',
            'confundido', 'perdido', 'difícil', 'complejo', 'imposible'
        }
        
        # Procesar texto
        processed_text = self.text_processor.preprocess_text(text, "basic")
        tokens = processed_text.split()
        
        positive_count = sum(1 for word in tokens if word in positive_words)
        negative_count = sum(1 for word in tokens if word in negative_words)
        
        total_tokens = len(tokens) if tokens else 1
        
        # Calcular score de sentimiento
        sentiment_score = (positive_count - negative_count) / total_tokens
        
        # Determinar etiqueta
        if sentiment_score > 0.1:
            label = "positive"
            emoji = "😊"
        elif sentiment_score < -0.1:
            label = "negative"
            emoji = "😞"
        else:
            label = "neutral"
            emoji = "😐"
        
        return {
            "score": sentiment_score,
            "label": label,
            "emoji": emoji,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "total_words": total_tokens
        }
    
    def get_conversation_insights(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Obtener insights de la conversación"""
        if not conversation_history:
            return {}
        
        # Análisis de mensajes
        user_messages = [msg["content"] for msg in conversation_history if msg.get("is_user")]
        bot_messages = [msg["content"] for msg in conversation_history if not msg.get("is_user")]
        
        # Sentimiento promedio
        user_sentiments = [self.analyze_sentiment(msg) for msg in user_messages]
        avg_sentiment = np.mean([s["score"] for s in user_sentiments]) if user_sentiments else 0
        
        # Entidades mencionadas
        all_entities = []
        for msg in user_messages:
            all_entities.extend(self.extract_entities(msg))
        
        # Estadísticas
        return {
            "total_messages": len(conversation_history),
            "user_message_count": len(user_messages),
            "bot_message_count": len(bot_messages),
            "average_sentiment": avg_sentiment,
            "entity_count": len(all_entities),
            "unique_entities": list(set([ent["text"] for ent in all_entities])),
            "conversation_duration": self._calculate_conversation_duration(conversation_history)
        }
    
    def _calculate_conversation_duration(self, conversation_history: List[Dict]) -> str:
        """Calcular duración de la conversación"""
        if len(conversation_history) < 2:
            return "0 minutos"
        
        first_message = conversation_history[0]
        last_message = conversation_history[-1]
        
        # Asumiendo que los mensajes tienen timestamp
        start_time = first_message.get("timestamp")
        end_time = last_message.get("timestamp")
        
        if start_time and end_time:
            # Cálculo simplificado de duración
            return "Varios minutos"
        
        return "Desconocido"

# Instancia global del motor NLP
nl_engine = NLEngine()