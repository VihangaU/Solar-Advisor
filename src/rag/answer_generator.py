"""Answer generation using Google Gemini LLM with retrieved documents"""
import re
import sys
import requests
import json
from typing import List
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class AnswerGenerator:
    """Generate clean, concise answers from retrieved chunks"""
    
    def __init__(self):
        """Initialize the answer generator with Google Gemini REST API"""
        config = Config()
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        self.api_key = config.GOOGLE_API_KEY
        self.model_name = config.LLM_MODEL  # "gemini-2.0-flash"
        self.temperature = config.LLM_TEMPERATURE

        # Ensure model name has models/ prefix for API call
        if not self.model_name.startswith("models/"):
            self.model_name = f"models/{self.model_name}"

        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:generateContent?key={self.api_key}"

        print(f"✓ Using Gemini REST API with model: {self.model_name}")

        # Define solar-related keywords
        self.solar_keywords = [
            # Core solar technology
            'solar', 'photovoltaic', 'pv', 'panel', 'cell', 'crystalline', 'monocrystalline',
            'polycrystalline', 'thin-film', 'bifacial', 'perovskite',

            # System components
            'inverter', 'micro-inverter', 'string inverter', 'charge controller', 'mppt', 'pwm',
            'battery', 'storage', 'combiner box', 'junction box', 'breaker', 'conduit',
            'racking', 'mounting', 'tracker', 'load controller',

            # Electrical terms
            'grid', 'net metering', 'on-grid', 'off-grid', 'hybrid', 'dc', 'ac', 'voltage',
            'ampere', 'kwh', 'kwp', 'watt', 'kilowatt', 'capacity', 'output', 'generation',
            'production', 'yield', 'ir drop', 'mismatch',

            # Performance & efficiency
            'efficiency', 'irradiance', 'sunlight', 'radiation', 'temperature coefficient',
            'degradation', 'soiling', 'shading', 'orientation', 'tilt', 'azimuth',

            # Installation & maintenance
            'installation', 'rooftop', 'mounting', 'wire', 'connection', 'maintenance',
            'warranty', 'inspection', 'cleaning',

            # Renewable energy related
            'renewable', 'energy', 'electricity', 'power', 'sun', 'clean energy', 'green energy',
            'alternative energy', 'sustainable', 'eco-friendly', 'environmental', 'carbon offset',
            'emissions', 'distributed generation',

            # Financial & incentives
            'cost', 'price', 'benefit', 'advantage', 'saving', 'save money', 'payback',
            'roi', 'irr', 'npv', 'financing', 'loan', 'grant', 'subsidy', 'incentive',
            'rebate', 'feed-in', 'tariff', 'electricity bill',

            # Sri Lanka specific
            'ceb', 'power cut', 'backup power', 'energy independence', 'blackout',
            'fuel cost', 'electricity tariff', 'sri lanka',

            # Installation context
            'residential', 'home', 'house', 'rooftop', 'setup', 'how much', 'install',
            'system', 'upgrade', 'retrofit'
        ]
    
    def is_query_relevant(self, query: str) -> bool:
        """
        Check if the query is related to solar energy
        
        Args:
            query: User's question
            
        Returns:
            True if solar-related, False otherwise
        """
        query_lower = query.lower()
        
        # Check for solar keywords in query
        keyword_count = sum(1 for keyword in self.solar_keywords if keyword in query_lower)
        
        # If at least 1 solar keyword found, consider it relevant
        if keyword_count >= 1:
            return True
        
        # Check for implicit solar queries (benefit, cost, install without explicit "solar")
        # These are only valid if they're about home energy systems
        implicit_terms = [
            'benefit', 'advantage', 'cost', 'price', 'install', 'setup',
            'how much', 'save money', 'electricity bill', 'power cut',
            'backup power', 'renewable', 'clean energy', 'green energy'
        ]
        
        has_implicit = any(term in query_lower for term in implicit_terms)
        has_home_context = any(word in query_lower for word in ['home', 'house', 'residential', 'rooftop'])
        
        # Only accept if both implicit term and home context are present
        if has_implicit and has_home_context:
            return True
        
        return False
    
    def is_retrieved_content_relevant(self, documents: List[str], query: str) -> bool:
        """
        Check if retrieved documents are actually about solar energy
        
        Args:
            documents: Retrieved document chunks
            query: User's question
            
        Returns:
            True if documents contain solar content, False otherwise
        """
        if not documents:
            return False
        
        # Combine all documents
        combined_text = ' '.join(documents).lower()
        
        # Count solar keywords in retrieved content
        keyword_count = sum(1 for keyword in self.solar_keywords if keyword in combined_text)

        # Need at least 2 solar keywords in retrieved content (lowered from 3 for better match)
        return keyword_count >= 2
    
    def clean_text(self, text: str) -> str:
        """Clean retrieved text by removing citations, URLs, and formatting"""
        # Remove citations like [1], [2], [1,2,3]
        text = re.sub(r'\[[\d,\s]+\]', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\[online\]\s*Available\s*:\s*\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Available\s*:\s*\S+', '', text, flags=re.IGNORECASE)
        
        # Remove file references
        text = re.sub(r'[\w\-]+\.pdf\s*\[\d+\]', '', text)
        text = re.sub(r'\[online\]', '', text, flags=re.IGNORECASE)
        
        # Remove page references
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        
        # Clean up multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_key_information(self, query: str, documents: List[str]) -> str:
        """Extract key information relevant to the query"""
        query_lower = query.lower()
        
        # Clean all documents first
        cleaned_docs = [self.clean_text(doc) for doc in documents]
        
        # Combine and split into sentences
        combined_text = ' '.join(cleaned_docs)
        sentences = re.split(r'[.!?]+', combined_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return combined_text
        
        # Extract query keywords (remove common words)
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'which', 'who', 'tell', 'me', 'about'}
        query_words = set(query_lower.split()) - stop_words
        
        # Score sentences based on relevance
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Count keyword matches
            matches = sum(1 for word in query_words if word in sentence_lower)
            
            # Prioritize sentences with numbers (for costs, specs)
            has_numbers = bool(re.search(r'\d+', sentence))
            
            # Prioritize sentences with key terms
            has_key_terms = any(term in sentence_lower for term in ['benefit', 'cost', 'price', 'advantage', 'save', 'efficient'])
            
            # Calculate score
            score = matches * 2
            if has_numbers:
                score += 1
            if has_key_terms:
                score += 1
            
            if score > 0:
                scored_sentences.append((score, sentence))
        
        if not scored_sentences:
            # Return first few sentences if no good matches
            return '. '.join(sentences[:3]) + '.'
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored_sentences[:5]]
        
        # Maintain original order
        result = []
        for sentence in sentences:
            if sentence in top_sentences:
                result.append(sentence)
        
        return '. '.join(result) + '.'
    
    def format_answer(self, answer: str, query: str) -> str:
        """Format the answer nicely with structure"""
        query_lower = query.lower()
        
        # For "what is" questions, keep it concise
        if any(phrase in query_lower for phrase in ['what is', 'what are', 'define']):
            # Take first 2-3 sentences
            sentences = re.split(r'[.!?]+', answer)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            return '. '.join(sentences[:2]) + '.'
        
        # For "how much" or cost questions, emphasize numbers
        if any(phrase in query_lower for phrase in ['how much', 'cost', 'price']):
            return answer
        
        # For benefit/advantage questions, structure as list if possible
        if any(phrase in query_lower for phrase in ['benefit', 'advantage', 'why']):
            # Try to detect list items
            if any(marker in answer for marker in ['•', '-', '1.', '2.', 'First', 'Second']):
                return answer
            
            # Split into points if answer is long
            sentences = re.split(r'[.!?]+', answer)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) > 3:
                # Format as numbered list
                formatted = []
                for i, sentence in enumerate(sentences[:5], 1):
                    formatted.append(f"{i}. {sentence}")
                return '\n'.join(formatted)
        
        return answer
    
    def _is_followup(self, query: str) -> bool:
        """Return True when the query looks like a follow-up to a previous answer."""
        q = query.lower().strip()
        followup_patterns = [
            'tell me more', 'explain more', 'elaborate', 'continue',
            'what about that', 'how about that', 'and the previous', 'give more details', 'give more information', 'can you expand on that', 'give me more', 'provide more', 'explain further',
            'mentioned above', 'you mentioned', 'as above', 'previous answer', 'previous response', 'more details', 'more information', 'can you expand', 'can you clarify', 'can you give more', 'can you provide more', 'can you explain further'
        ]
        return any(p in q for p in followup_patterns)

    def generate_answer(self, query: str, documents: List[str],
                        conversation_history: str = '') -> str:
        """
        Generate answer using Google Gemini LLM with retrieved documents as context.
        IMPORTANT: The LLM will ONLY use the provided documents to answer.

        Args:
            query: User's current question.
            documents: List of retrieved document chunks (from vector DB only).
            conversation_history: Optional prior Q&A turns for context.

        Returns:
            Generated answer or "not relevant" message
        """
        # First check: Is the query about solar energy?
        is_followup_with_history = bool(conversation_history) and self._is_followup(query)
        if not is_followup_with_history and not self.is_query_relevant(query):
            return "I'm sorry, but I can only answer questions related to solar energy systems, solar panels, installation, costs, and benefits in Sri Lanka. Please ask me about solar energy topics."

        # Second check: Are retrieved documents relevant?
        if not documents or not self.is_retrieved_content_relevant(documents, query):
            return "I don't have enough information to answer that specific question about solar energy. Please try rephrasing your question or ask about solar panels, costs, installation, benefits, or technical specifications."

        # Clean and prepare context from documents (ONLY SOURCE OF KNOWLEDGE)
        cleaned_docs = [self.clean_text(doc) for doc in documents]
        context = "\n\n".join(cleaned_docs)

        # Build the prompt with strict instructions
        system_instructions = """You are a helpful solar energy advisor for Sri Lanka.

IMPORTANT RULES:
1. Answer ONLY using the provided document context below
2. Do NOT use any external knowledge or information
3. If the answer is not in the provided documents, say "I don't have enough information"
4. Keep answers concise, clear, and factual
5. If asked about costs, always mention they may vary by location/installer
6. Be friendly but professional

DOCUMENT CONTEXT:
""" + context + "\n\n"

        # Build the full prompt
        if conversation_history and self._is_followup(query):
            full_prompt = system_instructions + f"""Previous conversation:
{conversation_history}

Current question: {query}

Please answer using ONLY the documents provided above."""
        else:
            full_prompt = system_instructions + f"""Question: {query}

Please answer using ONLY the documents provided above."""

        try:
            # Call Google Gemini REST API
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [{"text": full_prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": 500,
                }
            }

            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)

            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")

            result = response.json()

            if "candidates" in result and len(result["candidates"]) > 0:
                answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                return answer
            else:
                raise Exception("No response from API")

        except Exception as e:
            # Fallback: if API fails, use old extraction method
            print(f"⚠️ LLM generation failed: {str(e)}")
            if conversation_history and self._is_followup(query):
                enriched_query = f"{conversation_history}\n\nCurrent question: {query}"
                answer = self.extract_key_information(enriched_query, documents)
            else:
                answer = self.extract_key_information(query, documents)

            answer = self.clean_text(answer)
            answer = self.format_answer(answer, query)

            words = answer.split()
            if len(words) > 500:
                answer = ' '.join(words[:500]) + '...'

            return answer