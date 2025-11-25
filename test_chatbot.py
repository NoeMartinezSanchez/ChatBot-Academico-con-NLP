import json
from chatbot.utils.text_processor import text_processor

# Cargar intents
with open('data/intents.json', 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

available_intents = intents_data.get("intents", [])

print("🧪 TESTING CHATBOT MEJORADO")
print("=" * 50)

test_queries = [
    "eval",
    "evaluacion",
    "plataforma", 
    "como son las evaluaciones",
    "cómo acceder a la plataforma",
    "hola"
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    
    # Procesar texto
    enhanced = text_processor.enhance_query_understanding(query)
    print(f"   Texto mejorado: '{enhanced}'")
    
    # Buscar coincidencias manualmente
    best_intent = None
    best_score = 0
    
    for intent in available_intents:
        for pattern in intent["patterns"]:
            # Similitud básica
            pattern_lower = pattern.lower()
            query_lower = query.lower()
            
            # Coincidencia exacta
            if pattern_lower in query_lower or query_lower in pattern_lower:
                score = 0.9
            else:
                # Coincidencia de palabras
                pattern_words = set(text_processor.preprocess_text(pattern, "basic").split())
                query_words = set(text_processor.preprocess_text(query, "basic").split())
                common = pattern_words.intersection(query_words)
                score = len(common) / len(pattern_words) if pattern_words else 0
            
            if score > best_score:
                best_score = score
                best_intent = intent
    
    if best_intent and best_score > 0.3:
        print(f"   ✅ Intent detectado: {best_intent['tag']} (score: {best_score:.2f})")
    else:
        print(f"   ❌ No se detectó intent (mejor score: {best_score:.2f})")