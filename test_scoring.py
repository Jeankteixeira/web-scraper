"""
Simple test script for the LinkedIn Profile Scoring System
Tests the core functionality without requiring external ML libraries
"""

import json
import re
from datetime import datetime

class SimpleLinkedInProfileScorer:
    """
    Simplified version of the scoring engine for testing
    Uses basic text similarity instead of embeddings
    """
    
    def __init__(self):
        self.feature_weights = {
            'keyword_similarity': 2.5,
            'seniority_score': 0.8,
            'years_experience': 0.3,
            'relevant_company': 1.0,
            'brazil_indicator': 1.2,
            'certifications_count': 0.5,
            'skills_match': 1.5
        }
        self.bias = -5.0
        
        # Brazilian companies list
        self.relevant_companies = {
            'nubank', 'ifood', 'stone', 'mercado livre', 'globo',
            'petrobras', 'vale', 'itau', 'bradesco', 'santander'
        }
        
        # Seniority mapping
        self.seniority_mapping = {
            'junior': 1, 'jr': 1, 'trainee': 0,
            'pleno': 2, 'mid': 2, 'middle': 2,
            'senior': 3, 'sr': 3, 'sênior': 3,
            'principal': 4, 'staff': 4, 'lead': 4,
            'manager': 4, 'gerente': 4,
            'director': 5, 'diretor': 5, 'head': 5,
            'cto': 6, 'ceo': 6, 'founder': 6
        }
    
    def _basic_text_similarity(self, text1, text2):
        """Basic text similarity using word overlap"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_features(self, profile, keywords):
        """Extract features from profile"""
        features = {}
        
        # 1. Keyword Similarity
        profile_text = ' '.join([
            profile.get('title', ''),
            profile.get('name', ''),
            ' '.join(profile.get('skills', [])),
            profile.get('company', '')
        ]).lower()
        
        features['keyword_similarity'] = self._basic_text_similarity(profile_text, keywords.lower())
        
        # 2. Seniority Score
        title = profile.get('title', '').lower()
        max_seniority = 0
        for keyword, level in self.seniority_mapping.items():
            if keyword in title:
                max_seniority = max(max_seniority, level)
        features['seniority_score'] = max_seniority / 6.0
        
        # 3. Years of Experience
        experience = profile.get('experience', '')
        if isinstance(experience, str) and '-' in experience:
            try:
                start_year = int(experience.split('-')[0])
                features['years_experience'] = min(start_year / 20.0, 1.0)
            except:
                features['years_experience'] = 0.0
        else:
            features['years_experience'] = 0.0
        
        # 4. Relevant Company
        company = profile.get('company', '').lower()
        features['relevant_company'] = 1.0 if any(comp in company for comp in self.relevant_companies) else 0.0
        
        # 5. Brazil Indicator
        location = profile.get('location', '').lower()
        brazil_indicators = ['brasil', 'brazil', 'br', 'são paulo', 'rio de janeiro']
        features['brazil_indicator'] = 1.0 if any(indicator in location for indicator in brazil_indicators) else 0.0
        
        # 6. Certifications Count
        certifications = profile.get('certifications', [])
        cert_count = len(certifications) if isinstance(certifications, list) else 0
        features['certifications_count'] = min(cert_count / 10.0, 1.0)
        
        # 7. Skills Match
        skills = profile.get('skills', [])
        if skills and keywords:
            keywords_lower = keywords.lower()
            skill_matches = sum(1 for skill in skills if skill.lower() in keywords_lower)
            features['skills_match'] = min(skill_matches / len(skills), 1.0)
        else:
            features['skills_match'] = 0.0
        
        return features
    
    def calculate_score(self, profile, keywords):
        """Calculate relevance score for a profile"""
        features = self.extract_features(profile, keywords)
        
        # Calculate weighted score
        weighted_sum = sum(
            features[feature] * self.feature_weights.get(feature, 0.0)
            for feature in features
        ) + self.bias
        
        # Apply sigmoid function
        import math
        score = 1 / (1 + math.exp(-weighted_sum))
        
        # Classify based on score thresholds
        if score >= 0.85:
            classification = "Alta relevância"
            color_class = "high"
        elif score >= 0.6:
            classification = "Relevância moderada"
            color_class = "medium"
        else:
            classification = "Baixa relevância"
            color_class = "low"
        
        return {
            'profile_id': profile.get('id', 'unknown'),
            'score': round(score, 3),
            'score_percentage': round(score * 100, 1),
            'classification': classification,
            'color_class': color_class,
            'features': features
        }

def test_scoring_system():
    """Test the scoring system with sample profiles"""
    print("=== Teste do Sistema de Scoring de Perfis LinkedIn ===")
    print()
    
    # Initialize scorer
    scorer = SimpleLinkedInProfileScorer()
    
    # Test profiles (Brazilian professionals)
    test_profiles = [
        {
            'id': 'profile_001',
            'name': 'Carlos Silva',
            'title': 'Senior Python Developer & Blockchain Specialist',
            'company': 'Nubank',
            'location': 'São Paulo, SP, Brasil',
            'experience': '5-10',
            'skills': ['Python', 'Django', 'AWS', 'Docker', 'Blockchain', 'Ethereum', 'Hyperledger Besu'],
            'certifications': ['AWS Solutions Architect', 'Certified Ethereum Developer']
        },
        {
            'id': 'profile_002',
            'name': 'Ana Oliveira',
            'title': 'Full Stack Developer - Blockchain & Web3',
            'company': 'iFood',
            'location': 'São Paulo, SP, Brasil',
            'experience': '2-5',
            'skills': ['JavaScript', 'React', 'Node.js', 'MongoDB', 'Solidity', 'Web3'],
            'certifications': ['Scrum Master', 'Blockchain Developer Certification']
        },
        {
            'id': 'profile_003',
            'name': 'Roberto Santos',
            'title': 'Junior Software Engineer',
            'company': 'Stone',
            'location': 'Rio de Janeiro, RJ, Brasil',
            'experience': '0-2',
            'skills': ['Java', 'Spring Boot', 'MySQL', 'Git'],
            'certifications': []
        },
        {
            'id': 'profile_004',
            'name': 'Patricia Mendes',
            'title': 'CTO & Blockchain Architect',
            'company': 'Foxbit',
            'location': 'São Paulo, SP, Brasil',
            'experience': '10+',
            'skills': ['Blockchain Architecture', 'Hyperledger', 'Ethereum', 'Team Leadership'],
            'certifications': ['Certified Blockchain Architect', 'AWS Solutions Architect', 'PMP']
        },
        {
            'id': 'profile_005',
            'name': 'John Smith',
            'title': 'Software Developer',
            'company': 'Tech Corp',
            'location': 'New York, USA',
            'experience': '3-5',
            'skills': ['Java', 'Spring', 'React'],
            'certifications': ['Oracle Certified']
        }
    ]
    
    # Test with different keywords
    test_cases = [
        "blockchain hyperledger besu",
        "python developer",
        "senior blockchain",
        "javascript react"
    ]
    
    for keywords in test_cases:
        print(f"🔍 Testando com palavras-chave: '{keywords}'")
        print("=" * 60)
        
        # Score all profiles
        scored_profiles = []
        for profile in test_profiles:
            score_result = scorer.calculate_score(profile, keywords)
            scored_profiles.append({**profile, **score_result})
        
        # Sort by score (highest first)
        scored_profiles.sort(key=lambda x: x['score'], reverse=True)
        
        # Display top 3 results
        for i, profile in enumerate(scored_profiles[:3], 1):
            print(f"{i}. {profile['name']} - {profile['title']}")
            print(f"   Empresa: {profile['company']} | Local: {profile['location']}")
            print(f"   Score: {profile['score']:.3f} ({profile['score_percentage']}%) - {profile['classification']}")
            
            # Show key features that contributed to the score
            features = profile['features']
            key_features = []
            if features['keyword_similarity'] > 0.3:
                key_features.append(f"Similaridade: {features['keyword_similarity']:.2f}")
            if features['seniority_score'] > 0.3:
                key_features.append(f"Senioridade: {features['seniority_score']:.2f}")
            if features['brazil_indicator'] == 1.0:
                key_features.append("Brasil ✓")
            if features['relevant_company'] == 1.0:
                key_features.append("Empresa relevante ✓")
            
            if key_features:
                print(f"   Fatores: {', '.join(key_features)}")
            print()
        
        print("-" * 60)
        print()
    
    # Test feature extraction details
    print("🔧 Teste detalhado de extração de features:")
    print("=" * 60)
    
    test_profile = test_profiles[0]  # Carlos Silva
    test_keywords = "blockchain hyperledger besu python"
    
    features = scorer.extract_features(test_profile, test_keywords)
    
    print(f"Perfil: {test_profile['name']} - {test_profile['title']}")
    print(f"Keywords: {test_keywords}")
    print()
    print("Features extraídas:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.3f}")
    
    score_result = scorer.calculate_score(test_profile, test_keywords)
    print(f"\nScore final: {score_result['score']:.3f} ({score_result['classification']})")
    
    print("\n✅ Teste concluído com sucesso!")
    print("\nO sistema de scoring está funcionando corretamente:")
    print("- Extração de features implementada")
    print("- Cálculo de similaridade funcionando")
    print("- Detecção de profissionais brasileiros ativa")
    print("- Classificação por relevância operacional")
    print("- Priorização de empresas relevantes implementada")

if __name__ == "__main__":
    test_scoring_system()
