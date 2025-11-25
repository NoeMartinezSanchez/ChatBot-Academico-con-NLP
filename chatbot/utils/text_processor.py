import re
import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

class AdvancedTextProcessor:
    """
    Procesador de texto avanzado con NLP completo
    Combina spaCy, NLTK y scikit-learn
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self._load_nlp_models()
        self._setup_vectorizers()
        
    def _setup_logger(self):
        logger = logging.getLogger('TextProcessor')
        logger.setLevel(logging.INFO)
        return logger
    
    def _load_nlp_models(self):
        """Cargar modelos de NLP"""
        try:
            # Cargar modelo spaCy para español
            self.nlp = spacy.load("es_core_news_sm")
            self.logger.info("✅ Modelo spaCy cargado correctamente")
        except OSError as e:
            self.logger.error(f"❌ Error cargando spaCy: {e}")
            self.nlp = None
        
        # Configurar NLTK
        try:
            self.stop_words_es = set(stopwords.words('spanish'))
            self.stemmer = SnowballStemmer('spanish')
            self.lemmatizer = None  # Usaremos spaCy para lematización
            self.logger.info("✅ NLTK configurado correctamente")
        except LookupError as e:
            self.logger.error(f"❌ Error configurando NLTK: {e}")
            self.stop_words_es = set()
            self.stemmer = None
    
    def _setup_vectorizers(self):
        """Configurar vectorizadores para diferentes propósitos"""
        # Vectorizador TF-IDF para similitud semántica
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(self.stop_words_es),
            lowercase=True,
            ngram_range=(1, 2),  # Unigramas y bigramas
            min_df=2,
            max_df=0.8
        )
        
        # Vectorizador de conteo para características simples
        self.count_vectorizer = CountVectorizer(
            max_features=500,
            stop_words=list(self.stop_words_es)
        )
        
        self.is_tfidf_fitted = False
    
    def preprocess_text(self, text: str, method: str = "advanced") -> str:
        """
        Preprocesamiento de texto con diferentes niveles
        """
        if not text:
            return ""
        
        text = text.lower().strip()
        
        if method == "basic":
            return self._basic_preprocess(text)
        elif method == "advanced":
            return self._advanced_preprocess(text)
        elif method == "for_ml":
            return self._ml_preprocess(text)
        else:
            return text
    
    def _basic_preprocess(self, text: str) -> str:
        """Preprocesamiento básico: limpieza y normalización"""
        # Eliminar acentos
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Eliminar caracteres especiales, mantener letras, números y espacios
        text = re.sub(r'[^a-z0-9áéíóúñü\s]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _advanced_preprocess(self, text: str) -> str:
        """Preprocesamiento avanzado usando spaCy"""
        if not self.nlp:
            return self._basic_preprocess(text)
        
        doc = self.nlp(text)
        
        # Filtrar tokens: solo palabras alfabéticas, no stopwords, no puntuación
        tokens = [
            token.lemma_.lower() for token in doc
            if not token.is_stop 
            and not token.is_punct 
            and token.is_alpha
            and len(token.lemma_) > 2  # Filtrar palabras muy cortas
        ]
        
        return " ".join(tokens)
    
    def _ml_preprocess(self, text: str) -> str:
        """Preprocesamiento optimizado para ML"""
        processed = self._advanced_preprocess(text)
        
        # Aplicar stemming si está disponible
        if self.stemmer:
            tokens = processed.split()
            stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
            processed = " ".join(stemmed_tokens)
        
        return processed
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extraer características avanzadas del texto
        """
        features = {
            "basic": {
                "length": len(text),
                "word_count": len(text.split()),
                "char_count": len(text.replace(" ", "")),
                "has_question": "?" in text,
                "has_exclamation": "!" in text,
                "has_greeting": self._detect_greeting(text),
            },
            "nlp": {},
            "readability": {}
        }
        
        # Características NLP con spaCy
        if self.nlp:
            doc = self.nlp(text)
            nlp_features = self._extract_spacy_features(doc)
            features["nlp"].update(nlp_features)
        
        # Características de legibilidad
        readability_features = self._calculate_readability(text)
        features["readability"].update(readability_features)
        
        return features
    
    def _extract_spacy_features(self, doc) -> Dict[str, Any]:
        """Extraer características usando spaCy"""
        return {
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "entity_count": len(doc.ents),
            "pos_tags": [token.pos_ for token in doc],
            "unique_pos_tags": len(set([token.pos_ for token in doc])),
            "avg_word_length": np.mean([len(token.text) for token in doc if token.is_alpha]),
            "sentence_count": len(list(doc.sents)),
        }
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calcular métricas de legibilidad"""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return {"flesch_reading_ease": 0, "avg_sentence_length": 0}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Fórmula simplificada de legibilidad en español
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        
        return {
            "flesch_reading_ease": max(0, min(100, flesch_score)),
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
        }
    
    def _detect_greeting(self, text: str) -> bool:
        """Detectar si el texto contiene un saludo"""
        greetings = {
            'hola', 'buenos días', 'buenas tardes', 'buenas noches', 
            'saludos', 'hey', 'hi', 'qué tal', 'cómo estás'
        }
        text_lower = text.lower()
        return any(greeting in text_lower for greeting in greetings)


    def enhance_query_understanding(self, text: str) -> str:
        """Mejorar el entendimiento de consultas con variaciones"""
        text_lower = text.lower()
    
        # Normalizar variaciones comunes
        replacements = {
            'evaluacion': 'evaluación',
            'acceso': 'acceder',
            'registro': 'registrar',
            'plataforma': 'plataforma virtual',
            'estudio': 'estudiar',
            'como': 'cómo'
            }

        for wrong, correct in replacements.items():
            text_lower = text_lower.replace(wrong, correct)
    
        # Expandir términos abreviados
        term_expansions = {
            'eval': 'evaluación',
            'plat': 'plataforma',
            'tec': 'técnica',
            'est': 'estudio'
            }
        words = text_lower.split()
        expanded_words = []
        for word in words:
            expanded_words.append(term_expansions.get(word, word))
    
        return ' '.join(expanded_words)

    
    def get_word_embeddings(self, text: str) -> np.ndarray:
        """
        Obtener embeddings de palabras usando spaCy
        """
        if not self.nlp:
            return np.zeros(300)  # Vector cero si no hay modelo
        
        doc = self.nlp(text)
        
        # Promedio de embeddings de palabras relevantes
        valid_tokens = [token for token in doc if token.has_vector and not token.is_stop]
        
        if valid_tokens:
            embeddings = np.mean([token.vector for token in valid_tokens], axis=0)
        else:
            embeddings = np.zeros(300)  # Dimensión de embeddings de spaCy
        
        return embeddings
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calcular similitud semántica usando múltiples métodos
        """
        similarities = []
        
        # 1. Similitud con embeddings de spaCy
        if self.nlp:
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            spacy_similarity = doc1.similarity(doc2)
            similarities.append(spacy_similarity)
        
        # 2. Similitud TF-IDF
        try:
            if not self.is_tfidf_fitted:
                # Usar textos como corpus temporal
                self.tfidf_vectorizer.fit([text1, text2])
                self.is_tfidf_fitted = True
            
            vectors = self.tfidf_vectorizer.transform([text1, text2])
            tfidf_similarity = np.dot(vectors[0].toarray(), vectors[1].toarray().T)[0][0]
            similarities.append(tfidf_similarity)
        except Exception as e:
            self.logger.warning(f"Error en similitud TF-IDF: {e}")
        
        # 3. Similitud por Jaccard (palabras comunes)
        words1 = set(self._advanced_preprocess(text1).split())
        words2 = set(self._advanced_preprocess(text2).split())
        
        if words1 and words2:
            jaccard_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
            similarities.append(jaccard_similarity)
        
        # Promedio ponderado de similitudes
        if similarities:
            # Dar más peso a spaCy si está disponible
            weights = [0.5 if i == 0 and self.nlp else 0.25 for i in range(len(similarities))]
            total_weight = sum(weights)
            weighted_similarities = [s * w for s, w in zip(similarities, weights)]
            return sum(weighted_similarities) / total_weight
        else:
            return 0.0
    
    def fit_vectorizers(self, texts: List[str]):
        """Entrenar vectorizadores con un corpus de textos"""
        try:
            processed_texts = [self._ml_preprocess(text) for text in texts]
            self.tfidf_vectorizer.fit(processed_texts)
            self.count_vectorizer.fit(processed_texts)
            self.is_tfidf_fitted = True
            self.logger.info(f"✅ Vectorizadores entrenados con {len(texts)} textos")
        except Exception as e:
            self.logger.error(f"❌ Error entrenando vectorizadores: {e}")

# Instancia global del procesador
text_processor = AdvancedTextProcessor()