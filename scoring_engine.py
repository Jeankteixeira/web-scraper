import numpy as np
import re
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from config import LINKEDIN_CONFIG, SCORING_CONFIG
from utils import (
    calculate_text_similarity, extract_years_from_experience, 
    normalize_score, apply_sigmoid, create_timestamp, log_performance
)

# Try to import sentence-transformers, fallback to basic similarity if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using basic text similarity.")


class LinkedInProfileScorer:
    """
    LinkedIn Profile Scoring Engine for Brazilian professionals
    Uses ML-based features and logistic regression for relevance scoring
    
    Refactored for better performance and maintainability
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_weights = SCORING_CONFIG['feature_weights'].copy()
        self.bias = SCORING_CONFIG['bias']
        self.score_thresholds = SCORING_CONFIG['score_thresholds']
        self.classifications = SCORING_CONFIG['classifications']
        
        # Load configuration data
        self.relevant_companies = LINKEDIN_CONFIG['relevant_companies']
        self.seniority_mapping = LINKEDIN_CONFIG['seniority_mapping']
        
        # Initialize embedding model if available
        self.embedding_model = self._initialize_embedding_model()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _initialize_embedding_model(self) -> Optional[object]:
        """Initialize sentence transformer model with error handling"""
        if not EMBEDDINGS_AVAILABLE:
            return None
        
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.logger.info("Loaded multilingual embedding model for Portuguese/English support")
            return model
        except Exception as e:
            self.logger.warning(f"Could not load embedding model: {e}")
            return None
    
    def extract_features(self, profile: Dict, keywords: str) -> Dict[str, float]:
        """
        Extract and normalize features from a LinkedIn profile
        
        Args:
            profile: Dictionary containing profile information
            keywords: Search keywords for similarity calculation
            
        Returns:
            Dictionary of normalized features (0-1 scale)
        """
        features = {}
        
        # 1. Keyword Similarity (using embeddings or basic similarity)
        features['keyword_similarity'] = self._calculate_keyword_similarity(profile, keywords)
        
        # 2. Seniority Score (mapped to 0-1 scale)
        features['seniority_score'] = self._calculate_seniority_score(profile)
        
        # 3. Years of Experience (normalized)
        features['years_experience'] = self._normalize_experience(profile)
        
        # 4. Relevant Company Indicator (binary)
        features['relevant_company'] = self._check_relevant_company(profile)
        
        # 5. Brazil Location Indicator (binary with heuristics)
        features['brazil_indicator'] = self._check_brazil_location(profile)
        
        # 6. Certifications Count (normalized)
        features['certifications_count'] = self._normalize_certifications(profile)
        
        # 7. Skills Match Score
        features['skills_match'] = self._calculate_skills_match(profile, keywords)
        
        return features
    
    def _calculate_keyword_similarity(self, profile: Dict, keywords: str) -> float:
        """Calculate similarity between profile text and keywords using embeddings"""
        if not keywords.strip():
            return 0.0
        
        # Combine profile text fields
        profile_text = ' '.join([
            profile.get('title', ''),
            profile.get('name', ''),
            ' '.join(profile.get('skills', [])),
            profile.get('company', ''),
            ' '.join(profile.get('certifications', []))
        ]).lower()
        
        if not profile_text.strip():
            return 0.0
        
        # Use embeddings if available
        if self.embedding_model:
            try:
                profile_embedding = self.embedding_model.encode([profile_text])
                keywords_embedding = self.embedding_model.encode([keywords.lower()])
                similarity = cosine_similarity(profile_embedding, keywords_embedding)[0][0]
                return max(0.0, min(1.0, similarity))  # Ensure 0-1 range
            except Exception as e:
                self.logger.warning(f"Embedding calculation failed: {e}")
        
        # Fallback to basic text similarity
        return self._basic_text_similarity(profile_text, keywords.lower())
    
    def _basic_text_similarity(self, text1: str, text2: str) -> float:
        """Basic text similarity using utility function"""
        return calculate_text_similarity(text1, text2)
    
    def _calculate_seniority_score(self, profile: Dict) -> float:
        """Map seniority level to normalized score (0-1)"""
        title = profile.get('title', '').lower()
        
        # Check for seniority keywords in title
        max_seniority = 0
        for keyword, level in self.seniority_mapping.items():
            if keyword in title:
                max_seniority = max(max_seniority, level)
        
        # Normalize to 0-1 scale (max seniority is 6)
        return max_seniority / 6.0
    
    def _normalize_experience(self, profile: Dict) -> float:
        """Extract and normalize years of experience using utility function"""
        experience_data = extract_years_from_experience(profile.get('experience'))
        years_avg = experience_data.get('years_avg', 0)
        
        if years_avg:
            # Normalize to 0-1 scale (cap at 20 years)
            return normalize_score(years_avg / 20.0)
        
        return 0.0
    
    def _check_relevant_company(self, profile: Dict) -> float:
        """Check if profile is from a relevant company"""
        company = profile.get('company', '').lower()
        
        for relevant_company in self.relevant_companies:
            if relevant_company in company:
                return 1.0
        
        return 0.0
    
    def _check_brazil_location(self, profile: Dict) -> float:
        """Check if profile indicates Brazilian location using config patterns"""
        location = profile.get('location', '').lower()
        
        if not location:
            return 0.0
        
        # Use patterns from configuration
        for pattern in LINKEDIN_CONFIG['brazil_patterns']:
            if re.search(pattern, location, re.IGNORECASE):
                return 1.0
        
        return 0.0
    
    def _normalize_certifications(self, profile: Dict) -> float:
        """Normalize certification count"""
        certifications = profile.get('certifications', [])
        if isinstance(certifications, list):
            count = len(certifications)
        else:
            count = 0
        
        # Normalize to 0-1 scale (cap at 10 certifications)
        return min(count / 10.0, 1.0)
    
    def _calculate_skills_match(self, profile: Dict, keywords: str) -> float:
        """Calculate how well profile skills match keywords"""
        skills = profile.get('skills', [])
        if not skills or not keywords:
            return 0.0
        
        keywords_lower = keywords.lower()
        skill_matches = 0
        
        for skill in skills:
            if isinstance(skill, str) and skill.lower() in keywords_lower:
                skill_matches += 1
        
        # Normalize by total skills (cap at 1.0)
        return min(skill_matches / len(skills), 1.0) if skills else 0.0
    
    def calculate_score(self, profile: Dict, keywords: str) -> Dict:
        """
        Calculate relevance score for a profile
        
        Args:
            profile: Profile dictionary
            keywords: Search keywords
            
        Returns:
            Dictionary with score, classification, and feature breakdown
        """
        # Extract features
        features = self.extract_features(profile, keywords)
        
        # Calculate weighted score using logistic regression approach
        weighted_sum = sum(
            features[feature] * self.feature_weights.get(feature, 0.0)
            for feature in features
        ) + self.bias
        
        # Apply sigmoid function for 0-1 probability
        score = apply_sigmoid(weighted_sum)
        
        # Classify based on score thresholds from configuration
        if score >= self.score_thresholds['high']:
            classification = self.classifications['high']
            color_class = "high"
        elif score >= self.score_thresholds['medium']:
            classification = self.classifications['medium']
            color_class = "medium"
        else:
            classification = self.classifications['low']
            color_class = "low"
        
        # Log for audit
        self._log_scoring(profile, features, score, classification)
        
        return {
            'profile_id': profile.get('id', 'unknown'),
            'score': round(score, 3),
            'score_percentage': round(score * 100, 1),
            'classification': classification,
            'color_class': color_class,
            'features': features,
            'feature_explanations': self._generate_explanations(features)
        }
    
    def _generate_explanations(self, features: Dict[str, float]) -> List[str]:
        """Generate human-readable explanations for score factors"""
        explanations = []
        
        if features['keyword_similarity'] > 0.7:
            explanations.append(f"Alta similaridade com palavras-chave ({features['keyword_similarity']:.1%})")
        elif features['keyword_similarity'] > 0.4:
            explanations.append(f"Similaridade moderada com palavras-chave ({features['keyword_similarity']:.1%})")
        
        if features['seniority_score'] > 0.5:
            explanations.append(f"Nível de senioridade elevado ({features['seniority_score']:.1%})")
        
        if features['years_experience'] > 0.3:
            explanations.append(f"Experiência significativa ({features['years_experience']:.1%})")
        
        if features['relevant_company'] == 1.0:
            explanations.append("Trabalha em empresa relevante")
        
        if features['brazil_indicator'] == 1.0:
            explanations.append("Localizado no Brasil")
        
        if features['certifications_count'] > 0.2:
            explanations.append(f"Possui certificações ({features['certifications_count']:.1%})")
        
        if features['skills_match'] > 0.3:
            explanations.append(f"Skills relevantes ({features['skills_match']:.1%})")
        
        return explanations
    
    def _log_scoring(self, profile: Dict, features: Dict, score: float, classification: str):
        """Log scoring details for audit purposes"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'profile_id': profile.get('id', 'unknown'),
            'profile_name': profile.get('name', 'unknown'),
            'score': score,
            'classification': classification,
            'features': features
        }
        
        self.logger.info(f"Profile scored: {log_entry}")
    
    def batch_score_profiles(self, profiles: List[Dict], keywords: str) -> List[Dict]:
        """Score multiple profiles and return sorted by relevance"""
        scored_profiles = []
        
        for profile in profiles:
            try:
                score_result = self.calculate_score(profile, keywords)
                # Merge original profile with score data
                scored_profile = {**profile, **score_result}
                scored_profiles.append(scored_profile)
            except Exception as e:
                self.logger.error(f"Error scoring profile {profile.get('id', 'unknown')}: {e}")
                # Add profile with zero score if scoring fails
                scored_profile = {
                    **profile,
                    'score': 0.0,
                    'score_percentage': 0.0,
                    'classification': 'Erro na avaliação',
                    'color_class': 'error',
                    'features': {},
                    'feature_explanations': ['Erro no cálculo do score']
                }
                scored_profiles.append(scored_profile)
        
        # Sort by score (highest first)
        scored_profiles.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_profiles
    
    def update_weights(self, new_weights: Dict[str, float], new_bias: float = None):
        """Update feature weights for model tuning"""
        self.feature_weights.update(new_weights)
        if new_bias is not None:
            self.bias = new_bias
        
        self.logger.info(f"Updated weights: {self.feature_weights}")
        self.logger.info(f"Updated bias: {self.bias}")
    
    def get_model_info(self) -> Dict:
        """Get current model configuration"""
        return {
            'feature_weights': self.feature_weights,
            'bias': self.bias,
            'embeddings_available': EMBEDDINGS_AVAILABLE,
            'relevant_companies_count': len(self.relevant_companies),
            'seniority_levels': len(self.seniority_mapping)
        }
    
    def save_model(self, filepath: str):
        """Save model configuration to file"""
        model_data = {
            'feature_weights': self.feature_weights,
            'bias': self.bias,
            'relevant_companies': list(self.relevant_companies),
            'seniority_mapping': self.seniority_mapping
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model configuration from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.feature_weights = model_data.get('feature_weights', self.feature_weights)
            self.bias = model_data.get('bias', self.bias)
            
            if 'relevant_companies' in model_data:
                self.relevant_companies = set(model_data['relevant_companies'])
            
            if 'seniority_mapping' in model_data:
                self.seniority_mapping = model_data['seniority_mapping']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model from {filepath}: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize scorer
    scorer = LinkedInProfileScorer()
    
    # Test profile
    test_profile = {
        'id': 'test_001',
        'name': 'João Silva',
        'title': 'Senior Python Developer',
        'company': 'Nubank',
        'location': 'São Paulo, BR',
        'experience': '5-10',
        'skills': ['Python', 'Django', 'AWS', 'Docker'],
        'certifications': ['AWS Solutions Architect', 'Scrum Master']
    }
    
    # Test scoring
    keywords = "Python developer blockchain"
    result = scorer.calculate_score(test_profile, keywords)
    
    print("Scoring Result:")
    print(f"Score: {result['score']:.3f} ({result['score_percentage']}%)")
    print(f"Classification: {result['classification']}")
    print("Explanations:")
    for explanation in result['feature_explanations']:
        print(f"  - {explanation}")
    
    print("\nModel Info:")
    model_info = scorer.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
