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
        Procesar consulta - VERSIÓN SIMPLIFICADA Y FUNCIONAL
        """
        try:
            # Usar el texto mejorado
            enhanced_input = self.text_processor.enhance_query_understanding(user_input)
            processed_text = self.text_processor.preprocess_text(user_input, "advanced")
            features = self.text_processor.extract_features(user_input)
            
            print(f"🔍 DEBUG PROCESANDO: '{user_input}' -> '{enhanced_input}'")
            
            best_intent = None
            best_score = 0.0
            
            # Búsqueda directa y simple
            for intent in available_intents:
                intent_score = 0.0
                pattern_count = 0
                
                for pattern in intent["patterns"]:
                    pattern_lower = pattern.lower()
                    enhanced_lower = enhanced_input.lower()
                    user_lower = user_input.lower()
                    
                    # Estrategia 1: Coincidencia exacta
                    if pattern_lower in enhanced_lower or enhanced_lower in pattern_lower:
                        score = 0.9
                    elif pattern_lower in user_lower or user_lower in pattern_lower:
                        score = 0.8
                    else:
                        # Estrategia 2: Coincidencia de palabras
                        pattern_words = set(self.text_processor.preprocess_text(pattern, "basic").split())
                        enhanced_words = set(self.text_processor.preprocess_text(enhanced_input, "basic").split())
                        
                        if pattern_words and enhanced_words:
                            common_words = pattern_words.intersection(enhanced_words)
                            score = len(common_words) / len(pattern_words)
                        else:
                            score = 0
                    
                    # Bonus por términos clave específicos
                    key_terms = self._extract_key_terms(intent["tag"])
                    user_terms = set(enhanced_input.split())
                    common_terms = key_terms.intersection(user_terms)
                    
                    if common_terms:
                        score += len(common_terms) * 0.2
                    
                    intent_score += score
                    pattern_count += 1
                    
                    if score > best_score:
                        best_score = score
                        best_intent = intent
                
                # También considerar el promedio por intent
                if pattern_count > 0:
                    avg_score = intent_score / pattern_count
                    if avg_score > best_score:
                        best_score = avg_score
                        best_intent = intent
            
            # Reglas especiales para casos comunes
            enhanced_lower = enhanced_input.lower()
            if any(word in enhanced_lower for word in ['hola', 'buenos', 'buenas', 'hi', 'hey']):
                best_intent = next((i for i in available_intents if i["tag"] == "saludo"), best_intent)
                best_score = max(best_score, 0.9)
            
            if any(word in enhanced_lower for word in ['adiós', 'chao', 'bye', 'hasta luego']):
                best_intent = next((i for i in available_intents if i["tag"] == "despedida"), best_intent)
                best_score = max(best_score, 0.9)
            
            print(f"🎯 RESULTADO: {best_intent['tag'] if best_intent else 'None'} (score: {best_score:.2f})")
            
            return {
                "user_input": user_input,
                "processed_text": processed_text,
                "features": features,
                "final_intent": best_intent,
                "confidence": best_score,
                "winning_method": "enhanced_semantic"
            }
        
        except Exception as e:
            self.logger.error(f"Error en process_query: {e}")
            # Fallback básico
            return {
                "user_input": user_input,
                "processed_text": user_input,
                "features": {},
                "final_intent": None,
                "confidence": 0.0,
                "winning_method": "error"
            }
    
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
            
            # Mapear a tags de intents
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
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error en clasificación ML: {e}")
            return {"method": "ml", "available": False, "intent": None, "confidence": 0.0}
    
    def _classify_with_semantic_similarity(self, user_input: str, intents: List[Dict]) -> Dict[str, Any]:
        """Clasificación MEJORADA con entendimiento contextual"""
        best_intent = None
        best_score = 0.0
        best_pattern = ""
        
        # Preprocesamiento mejorado con el nuevo método
        enhanced_input = self.text_processor.enhance_query_understanding(user_input)
        
        for intent in intents:
            intent_score = 0.0
            pattern_count = 0
            
            for pattern in intent["patterns"]:
                # Calcular similitud con el texto original
                similarity1 = self.text_processor.calculate_semantic_similarity(user_input, pattern)
                
                # Calcular similitud con el texto mejorado
                similarity2 = self.text_processor.calculate_semantic_similarity(enhanced_input, pattern)
                
                # Usar la mejor similitud
                similarity = max(similarity1, similarity2)
                
                # Bonus por coincidencia de palabras clave importantes
                key_terms = self._extract_key_terms(intent["tag"])
                user_terms = set(enhanced_input.split())
                common_terms = key_terms.intersection(user_terms)
                
                if common_terms:
                    similarity += len(common_terms) * 0.1
                
                intent_score += similarity
                pattern_count += 1
                
                if similarity > best_score:
                    best_score = similarity
                    best_intent = intent
                    best_pattern = pattern
            
            # También considerar el promedio por intent
            if pattern_count > 0:
                avg_score = intent_score / pattern_count
                if avg_score > best_score:
                    best_score = avg_score
                    best_intent = intent
        
        return {
            "method": "semantic",
            "intent": best_intent,
            "confidence": best_score,
            "matched_pattern": best_pattern
        }
    
    def _extract_key_terms(self, intent_tag: str) -> set:
        """Extraer términos clave por tipo de intent"""
        key_terms = {
            "saludo": {"hola", "buenos", "buenas", "saludos", "hi", "hey"},
            "despedida": {"adiós", "chao", "bye", "hasta", "luego", "nos vemos"},
            "modulo_info": {"módulo", "propedéutico", "información", "qué es", "para qué"},
            "duracion_modulo": {"duración", "cuánto", "dura", "semanas", "tiempo"},
            "evaluacion": {"evaluación", "calificación", "calificar", "nota", "aprob", "exam", "eval"},
            "plataforma_acceso": {"plataforma", "acceso", "entrar", "login", "registro", "ingresar", "plat"},
            "soporte_tecnico": {"soporte", "problema", "error", "técnico", "ayuda", "falla"},
            "tecnicas_estudio_basicas": {"técnica", "estudio", "estudiar", "método", "aprender", "tec", "est"},
            "tecnicas_estudio_avanzadas": {"avanzada", "eficiente", "comprobada", "científica", "efectiva"},
            "tecnica_pomodoro": {"pomodoro", "tomate", "25 minutos", "técnica"},
            "mapas_mentales": {"mapa", "mental", "visual", "diagrama", "organizar"},
            "organizacion_tiempo": {"organización", "tiempo", "horario", "planificación", "gestión"},
            "foros_participacion": {"foro", "participación", "discusión", "contribuir", "comentar"}
        }
        return key_terms.get(intent_tag, set())
    
    def _classify_with_rules(self, user_input: str, intents: List[Dict], features: Dict) -> Dict[str, Any]:
        """Clasificación MEJORADA usando reglas y patrones"""
        # Usar el texto mejorado para mejor detección
        enhanced_input = self.text_processor.enhance_query_understanding(user_input)
        user_input_lower = enhanced_input.lower()
        
        best_intent = None
        best_score = 0.0
        
        for intent in intents:
            score = 0.0
            
            # Verificar patrones simples con texto mejorado
            for pattern in intent["patterns"]:
                pattern_lower = pattern.lower()
                
                # Coincidencia exacta con texto mejorado
                if pattern_lower in user_input_lower:
                    score = max(score, 0.8)
                
                # Coincidencia de palabras clave mejorada
                pattern_words = set(self.text_processor.preprocess_text(pattern, "basic").split())
                user_words = set(self.text_processor.preprocess_text(enhanced_input, "basic").split())
                
                common_words = pattern_words.intersection(user_words)
                if common_words:
                    word_score = len(common_words) / len(pattern_words) * 0.6
                    score = max(score, word_score)
            
            # Reglas contextuales mejoradas
            if features["basic"]["has_greeting"] and intent["tag"] == "saludo":
                score = max(score, 0.9)
            
            if features["basic"]["has_question"] and intent["tag"] in ["evaluacion", "modulo_info", "plataforma_acceso"]:
                score = max(score, 0.3)
            
            # Bonus por términos clave específicos
            key_terms = self._extract_key_terms(intent["tag"])
            user_terms = set(user_input_lower.split())
            common_key_terms = key_terms.intersection(user_terms)
            
            if common_key_terms:
                score += len(common_key_terms) * 0.15
            
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
            "winning_method": winning_method
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
                    "end": ent.end_char
                })
            
            return entities
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo entidades: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Análisis de sentimiento básico"""
        # Palabras positivas y negativas en español
        positive_words = {
            'bueno', 'excelente', 'genial', 'perfecto', 'ayuda', 'gracias', 'bien',
            'fantástico', 'maravilloso', 'útil', 'agradecido'
        }
        
        negative_words = {
            'malo', 'problema', 'error', 'difícil', 'complicado', 'no', 'nunca',
            'terrible', 'horrible', 'pésimo', 'frustrado'
        }
        
        processed_text = self.text_processor.preprocess_text(text, "basic")
        tokens = processed_text.split()
        
        positive_count = sum(1 for word in tokens if word in positive_words)
        negative_count = sum(1 for word in tokens if word in negative_words)
        
        total_tokens = len(tokens) if tokens else 1
        sentiment_score = (positive_count - negative_count) / total_tokens
        
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
            "negative_words": negative_count
        }

# Instancia global del motor NLP
nl_engine = NLEngine()