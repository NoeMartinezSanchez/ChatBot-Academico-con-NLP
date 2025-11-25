import random
import logging
from typing import Dict, Any, List
from datetime import datetime

class ResponseGenerator:
    def __init__(self, available_intents: List[Dict]):
        self.available_intents = available_intents
        self.logger = self.setup_logger()
        self.conversation_context = {}
        
        # Respuestas contextuales mejoradas
        self.default_responses = {
            "general": [
                "🤔 No estoy seguro de entender completamente. ¿Podrías reformular tu pregunta?",
                "💭 Interesante consulta. Puedo ayudarte con información del módulo propedéutico, técnicas de estudio, plataforma virtual y más.",
                "🎓 Como asistente especializado, puedo orientarte sobre el módulo propedéutico y tu adaptación académica."
            ],
            "high_confidence": [
                "¿Te ha quedado claro? ¿Necesitas más información sobre este tema?",
                "Espero que esta información te sea útil. ¿Tienes alguna otra pregunta?",
                "¿Hay algo más específico que te gustaría saber sobre esto?"
            ],
            "low_confidence": [
                "¿Es esto lo que necesitabas? Si no, por favor reformula tu pregunta.",
                "Creo que esto podría ayudarte. ¿Responde a tu duda?",
                "¿Esta información aborda tu consulta? Si no, intenta preguntar de otra forma."
            ]
        }
    
    def setup_logger(self):
        logger = logging.getLogger('ResponseGenerator')
        logger.setLevel(logging.INFO)
        return logger
    
    def get_response(self, nlp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generar respuesta basada en resultados NLP"""
        try:
            user_input = nlp_results["user_input"]
            final_intent = nlp_results["final_intent"]
            confidence = nlp_results["confidence"]
            features = nlp_results["features"]
            
            # Determinar contexto de respuesta
            context = self._determine_context(final_intent, features, confidence)
            
            if final_intent and confidence > 0.6:
                response_data = self._get_intent_response(final_intent, confidence, context)
            else:
                response_data = self._get_fallback_response(nlp_results, context)
            
            # Agregar metadata de NLP
            response_data.update({
                "nlp_confidence": confidence,
                "detected_intent": final_intent["tag"] if final_intent else "unknown",
                "classification_method": nlp_results["winning_method"],
                "context": context,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Respuesta generada - Intent: {response_data['detected_intent']}, "
                           f"Conf: {confidence:.3f}, Método: {nlp_results['winning_method']}")
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error generando respuesta: {e}")
            return self._get_error_response()
    
    def _determine_context(self, intent: Dict, features: Dict, confidence: float) -> str:
        """Determinar el contexto de la respuesta"""
        if not intent:
            return "unknown"
        
        intent_tag = intent.get("tag", "")
        
        # Contextos basados en el tipo de intent
        if intent_tag == "saludo":
            return "welcome"
        elif intent_tag == "despedida":
            return "goodbye"
        elif confidence > 0.8:
            return "high_confidence"
        elif confidence > 0.5:
            return "medium_confidence"
        else:
            return "low_confidence"
    
    def _get_intent_response(self, intent: Dict, confidence: float, context: str) -> Dict[str, Any]:
        """Generar respuesta para intent reconocido"""
        # Seleccionar respuesta aleatoria del intent
        response = random.choice(intent["responses"])
        
        # Mejorar respuesta basada en confianza y contexto
        if context == "low_confidence":
            clarification = random.choice(self.default_responses["low_confidence"])
            response = f"{response}\n\n{clarification}"
        elif context == "high_confidence":
            follow_up = random.choice(self.default_responses["high_confidence"])
            response = f"{response}\n\n{follow_up}"
        
        # Agregar emojis basados en el tipo de intent
        response = self._add_contextual_emoji(response, intent["tag"])
        
        return {
            "response": response,
            "status": "success",
            "has_suggestions": context in ["low_confidence", "unknown"],
            "intent_tag": intent["tag"],
            "context": context
        }
    
    def _get_fallback_response(self, nlp_results: Dict, context: str) -> Dict[str, Any]:
        """Generar respuesta cuando no se reconoce el intent"""
        base_response = random.choice(self.default_responses["general"])
        
        # Agregar sugerencias basadas en análisis NLP
        suggestions = self._generate_smart_suggestions(nlp_results)
        
        response = f"{base_response}\n\n{suggestions}"
        
        return {
            "response": response,
            "status": "unknown_intent",
            "has_suggestions": True,
            "intent_tag": "unknown",
            "context": context
        }
    
    def _generate_smart_suggestions(self, nlp_results: Dict) -> str:
        """Generar sugerencias inteligentes basadas en análisis NLP"""
        features = nlp_results["features"]
        entities = nlp_results.get("entities", [])
        
        suggestions = []
        
        # Sugerir basado en entidades detectadas
        if entities:
            entity_types = set(entity["label"] for entity in entities)
            if "ORG" in entity_types:
                suggestions.append("¿Necesitas información sobre **organizaciones o instituciones**?")
        
        # Sugerir basado en características del texto
        if features["basic"]["has_question"]:
            suggestions.append("Puedo responder preguntas sobre **técnicas de estudio, evaluación o plataforma**.")
        
        if features["basic"]["has_greeting"]:
            suggestions.append("¡Puedo saludarte y ayudarte con información del módulo propedéutico!")
        
        # Sugerencias generales si no hay contexto específico
        if not suggestions:
            general_suggestions = [
                "💡 **Sugerencia:** ¿Te interesa saber sobre las **técnicas de estudio** como Pomodoro?",
                "💡 **Sugerencia:** ¿Necesitas ayuda con el **acceso a la plataforma** virtual?",
                "💡 **Sugerencia:** ¿Quieres información sobre la **evaluación del módulo**?",
                "💡 **Sugerencia:** ¿Buscas consejos de **organización del tiempo**?"
            ]
            suggestions.append(random.choice(general_suggestions))
        
        return "\n".join(suggestions[:2])  # Máximo 2 sugerencias
    
    def _add_contextual_emoji(self, response: str, intent_tag: str) -> str:
        """Agregar emojis contextuales a la respuesta"""
        emoji_map = {
            "saludo": "👋",
            "despedida": "👋",
            "modulo_info": "📚",
            "tecnicas_estudio": "🎯",
            "tecnica_pomodoro": "🍅",
            "mapas_mentales": "🧠",
            "plataforma_acceso": "💻",
            "soporte_tecnico": "🛠️",
            "evaluacion": "📊",
            "organizacion_tiempo": "⏰",
            "foros_participacion": "💬"
        }
        
        emoji = emoji_map.get(intent_tag, "🤖")
        return f"{emoji} {response}"
    
    def _get_error_response(self) -> Dict[str, Any]:
        """Generar respuesta de error"""
        return {
            "response": "❌ Lo siento, he tenido un problema interno. Por favor, intenta nuevamente o reformula tu pregunta.",
            "status": "error",
            "has_suggestions": False,
            "intent_tag": "error",
            "context": "error"
        }
    
    def get_available_topics(self) -> List[str]:
        """Obtener lista de temas disponibles para sugerencias"""
        topics = set()
        for intent in self.available_intents:
            topics.add(intent["tag"])
            if "context" in intent:
                topics.add(intent["context"])
        return sorted(list(topics))