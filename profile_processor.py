import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

from config import LINKEDIN_CONFIG, TECH_CATEGORIES
from utils import (
    clean_text, standardize_list_field, extract_years_from_experience,
    create_timestamp, validate_profile_data
)

class ProfileProcessor:
    """
    Profile data processor for LinkedIn profiles
    Handles data cleaning, normalization, and feature engineering
    
    Refactored to use centralized configuration and utility functions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration data
        self.brazil_patterns = LINKEDIN_CONFIG['brazil_patterns']
        self.tech_categories = TECH_CATEGORIES
        self.seniority_mapping = LINKEDIN_CONFIG['seniority_mapping']
        
        # Experience level indicators (derived from seniority mapping)
        self.experience_indicators = self._build_experience_indicators()
    
    def _build_experience_indicators(self) -> Dict[str, List[str]]:
        """Build experience indicators from seniority mapping"""
        indicators = {
            'junior': [],
            'mid': [],
            'senior': [],
            'management': [],
            'executive': []
        }
        
        for keyword, level in self.seniority_mapping.items():
            if level <= 1:
                indicators['junior'].append(keyword)
            elif level == 2:
                indicators['mid'].append(keyword)
            elif level == 3:
                indicators['senior'].append(keyword)
            elif level == 4:
                indicators['management'].append(keyword)
            else:
                indicators['executive'].append(keyword)
        
        return indicators
    
    def clean_profile_data(self, profile: Dict) -> Dict:
        """
        Clean and standardize profile data
        
        Args:
            profile: Raw profile dictionary
            
        Returns:
            Cleaned profile dictionary
        """
        cleaned_profile = profile.copy()
        
        # Clean text fields
        text_fields = ['name', 'title', 'company', 'location']
        for field in text_fields:
            if field in cleaned_profile and cleaned_profile[field]:
                cleaned_profile[field] = self._clean_text(cleaned_profile[field])
        
        # Standardize skills list
        if 'skills' in cleaned_profile:
            cleaned_profile['skills'] = self._standardize_skills(cleaned_profile['skills'])
        
        # Standardize certifications list
        if 'certifications' in cleaned_profile:
            cleaned_profile['certifications'] = self._standardize_certifications(cleaned_profile['certifications'])
        
        # Normalize experience field
        if 'experience' in cleaned_profile:
            cleaned_profile['experience_normalized'] = self._normalize_experience_field(cleaned_profile['experience'])
        
        # Add derived fields
        cleaned_profile['brazil_score'] = self._calculate_brazil_score(cleaned_profile)
        cleaned_profile['tech_category_scores'] = self._calculate_tech_category_scores(cleaned_profile)
        cleaned_profile['experience_level'] = self._determine_experience_level(cleaned_profile)
        
        return cleaned_profile
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text fields"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep accents for Portuguese
        text = re.sub(r'[^\w\s\-\.\,\(\)\/\&\+]', '', text)
        
        return text
    
    def _standardize_skills(self, skills) -> List[str]:
        """Standardize skills list"""
        if not skills:
            return []
        
        if isinstance(skills, str):
            # Split by common delimiters
            skills = re.split(r'[,;|]', skills)
        
        standardized_skills = []
        for skill in skills:
            if isinstance(skill, str):
                skill = skill.strip().lower()
                if skill and len(skill) > 1:  # Filter out single characters
                    standardized_skills.append(skill)
        
        return list(set(standardized_skills))  # Remove duplicates
    
    def _standardize_certifications(self, certifications) -> List[str]:
        """Standardize certifications list"""
        if not certifications:
            return []
        
        if isinstance(certifications, str):
            certifications = re.split(r'[,;|]', certifications)
        
        standardized_certs = []
        for cert in certifications:
            if isinstance(cert, str):
                cert = cert.strip()
                if cert and len(cert) > 2:  # Filter out very short strings
                    standardized_certs.append(cert)
        
        return list(set(standardized_certs))  # Remove duplicates
    
    def _normalize_experience_field(self, experience) -> Dict[str, Optional[int]]:
        """Extract and normalize experience information"""
        result = {
            'years_min': None,
            'years_max': None,
            'years_avg': None
        }
        
        if not experience:
            return result
        
        experience_str = str(experience).lower()
        
        # Pattern for ranges like "5-10", "2-5 years", etc.
        range_pattern = r'(\d+)\s*[-–]\s*(\d+)'
        range_match = re.search(range_pattern, experience_str)
        
        if range_match:
            min_years = int(range_match.group(1))
            max_years = int(range_match.group(2))
            result['years_min'] = min_years
            result['years_max'] = max_years
            result['years_avg'] = (min_years + max_years) / 2
        else:
            # Pattern for single numbers like "5 years", "10+ years"
            single_pattern = r'(\d+)\+?'
            single_match = re.search(single_pattern, experience_str)
            
            if single_match:
                years = int(single_match.group(1))
                result['years_min'] = years
                result['years_max'] = years
                result['years_avg'] = years
        
        return result
    
    def _calculate_brazil_score(self, profile: Dict) -> float:
        """Calculate Brazil location confidence score"""
        score = 0.0
        location = profile.get('location', '').lower()
        
        if not location:
            return 0.0
        
        # Check for Brazil patterns
        for pattern in self.brazil_patterns:
            if re.search(pattern, location, re.IGNORECASE):
                score += 0.3
        
        # Additional heuristics
        if 'br' in location and len(location) < 20:  # Short location with BR
            score += 0.4
        
        if any(city in location for city in ['são paulo', 'rio de janeiro', 'belo horizonte']):
            score += 0.5  # Major Brazilian cities
        
        # Check company location (if Brazilian company)
        company = profile.get('company', '').lower()
        brazilian_companies = ['nubank', 'ifood', 'stone', 'mercado livre', 'globo', 'petrobras']
        if any(comp in company for comp in brazilian_companies):
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_tech_category_scores(self, profile: Dict) -> Dict[str, float]:
        """Calculate scores for different technology categories"""
        skills = profile.get('skills', [])
        if not skills:
            return {category: 0.0 for category in self.tech_categories}
        
        category_scores = {}
        
        for category, tech_list in self.tech_categories.items():
            matches = 0
            for skill in skills:
                skill_lower = skill.lower()
                for tech in tech_list:
                    if tech in skill_lower or skill_lower in tech:
                        matches += 1
                        break  # Count each skill only once per category
            
            # Normalize by category size
            category_scores[category] = min(matches / len(tech_list), 1.0)
        
        return category_scores
    
    def _determine_experience_level(self, profile: Dict) -> str:
        """Determine experience level based on title and experience"""
        title = profile.get('title', '').lower()
        experience_data = profile.get('experience_normalized', {})
        years_avg = experience_data.get('years_avg', 0)
        
        # Check title for experience indicators
        for level, indicators in self.experience_indicators.items():
            for indicator in indicators:
                if indicator in title:
                    return level
        
        # Fallback to years of experience
        if years_avg:
            if years_avg < 2:
                return 'junior'
            elif years_avg < 5:
                return 'mid'
            elif years_avg < 10:
                return 'senior'
            else:
                return 'executive'
        
        return 'unknown'
    
    def enhance_profile_for_scoring(self, profile: Dict, keywords: str = "") -> Dict:
        """
        Enhance profile with additional features for scoring
        
        Args:
            profile: Cleaned profile dictionary
            keywords: Search keywords for context
            
        Returns:
            Enhanced profile with scoring features
        """
        enhanced_profile = profile.copy()
        
        # Add keyword relevance features
        if keywords:
            enhanced_profile['keyword_features'] = self._extract_keyword_features(profile, keywords)
        
        # Add quality indicators
        enhanced_profile['quality_score'] = self._calculate_quality_score(profile)
        
        # Add completeness score
        enhanced_profile['completeness_score'] = self._calculate_completeness_score(profile)
        
        # Add professional network indicators
        enhanced_profile['network_indicators'] = self._extract_network_indicators(profile)
        
        return enhanced_profile
    
    def _extract_keyword_features(self, profile: Dict, keywords: str) -> Dict:
        """Extract keyword-related features"""
        keywords_lower = keywords.lower()
        keyword_tokens = set(re.findall(r'\w+', keywords_lower))
        
        features = {
            'title_keyword_matches': 0,
            'skills_keyword_matches': 0,
            'company_keyword_matches': 0,
            'total_keyword_density': 0.0
        }
        
        # Title matches
        title = profile.get('title', '').lower()
        title_tokens = set(re.findall(r'\w+', title))
        features['title_keyword_matches'] = len(keyword_tokens.intersection(title_tokens))
        
        # Skills matches
        skills = profile.get('skills', [])
        skills_text = ' '.join(skills).lower()
        skills_tokens = set(re.findall(r'\w+', skills_text))
        features['skills_keyword_matches'] = len(keyword_tokens.intersection(skills_tokens))
        
        # Company matches
        company = profile.get('company', '').lower()
        company_tokens = set(re.findall(r'\w+', company))
        features['company_keyword_matches'] = len(keyword_tokens.intersection(company_tokens))
        
        # Overall keyword density
        all_profile_text = f"{title} {skills_text} {company}".lower()
        all_tokens = re.findall(r'\w+', all_profile_text)
        if all_tokens:
            keyword_count = sum(1 for token in all_tokens if token in keyword_tokens)
            features['total_keyword_density'] = keyword_count / len(all_tokens)
        
        return features
    
    def _calculate_quality_score(self, profile: Dict) -> float:
        """Calculate overall profile quality score"""
        score = 0.0
        
        # Check for professional photo (placeholder logic)
        if profile.get('photo'):
            score += 0.1
        
        # Check title quality
        title = profile.get('title', '')
        if title and len(title) > 10:
            score += 0.2
        
        # Check skills count
        skills = profile.get('skills', [])
        if len(skills) >= 5:
            score += 0.2
        elif len(skills) >= 3:
            score += 0.1
        
        # Check certifications
        certifications = profile.get('certifications', [])
        if len(certifications) >= 2:
            score += 0.2
        elif len(certifications) >= 1:
            score += 0.1
        
        # Check experience information
        if profile.get('experience_normalized', {}).get('years_avg'):
            score += 0.2
        
        # Check location specificity
        location = profile.get('location', '')
        if location and len(location) > 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_completeness_score(self, profile: Dict) -> float:
        """Calculate profile completeness score"""
        required_fields = ['name', 'title', 'company', 'location']
        optional_fields = ['skills', 'certifications', 'experience']
        
        score = 0.0
        
        # Required fields (60% of score)
        for field in required_fields:
            if profile.get(field):
                score += 0.15
        
        # Optional fields (40% of score)
        for field in optional_fields:
            if profile.get(field):
                score += 0.133
        
        return min(score, 1.0)
    
    def _extract_network_indicators(self, profile: Dict) -> Dict:
        """Extract professional network indicators"""
        indicators = {
            'has_linkedin_url': bool(profile.get('link') or profile.get('url')),
            'company_size_indicator': self._estimate_company_size(profile.get('company', '')),
            'title_seniority_indicator': self._extract_title_seniority(profile.get('title', ''))
        }
        
        return indicators
    
    def _estimate_company_size(self, company: str) -> str:
        """Estimate company size based on company name"""
        if not company:
            return 'unknown'
        
        company_lower = company.lower()
        
        # Large companies
        large_companies = [
            'google', 'microsoft', 'amazon', 'apple', 'meta', 'netflix',
            'uber', 'airbnb', 'spotify', 'linkedin', 'twitter', 'tesla',
            'nubank', 'ifood', 'mercado livre', 'globo', 'petrobras', 'vale'
        ]
        
        if any(comp in company_lower for comp in large_companies):
            return 'large'
        
        # Indicators of medium companies
        if any(indicator in company_lower for indicator in ['ltda', 'inc', 'corp', 'sa']):
            return 'medium'
        
        return 'small'
    
    def _extract_title_seniority(self, title: str) -> int:
        """Extract numeric seniority level from title"""
        if not title:
            return 0
        
        title_lower = title.lower()
        
        # Executive level (5)
        if any(word in title_lower for word in ['ceo', 'cto', 'founder', 'president', 'vp']):
            return 5
        
        # Director level (4)
        if any(word in title_lower for word in ['director', 'diretor', 'head']):
            return 4
        
        # Manager level (3)
        if any(word in title_lower for word in ['manager', 'gerente', 'lead', 'principal']):
            return 3
        
        # Senior level (2)
        if any(word in title_lower for word in ['senior', 'sr', 'sênior']):
            return 2
        
        # Junior level (1)
        if any(word in title_lower for word in ['junior', 'jr', 'trainee', 'intern']):
            return 1
        
        return 1  # Default to junior if no indicators found
    
    def batch_process_profiles(self, profiles: List[Dict], keywords: str = "") -> List[Dict]:
        """
        Process multiple profiles in batch
        
        Args:
            profiles: List of raw profile dictionaries
            keywords: Search keywords for context
            
        Returns:
            List of processed and enhanced profiles
        """
        processed_profiles = []
        
        for i, profile in enumerate(profiles):
            try:
                # Clean the profile
                cleaned_profile = self.clean_profile_data(profile)
                
                # Enhance for scoring
                enhanced_profile = self.enhance_profile_for_scoring(cleaned_profile, keywords)
                
                # Add processing metadata
                enhanced_profile['processing_metadata'] = {
                    'processed_at': datetime.now().isoformat(),
                    'processor_version': '1.0',
                    'batch_index': i
                }
                
                processed_profiles.append(enhanced_profile)
                
            except Exception as e:
                self.logger.error(f"Error processing profile {i}: {e}")
                # Add original profile with error flag
                error_profile = profile.copy()
                error_profile['processing_error'] = str(e)
                processed_profiles.append(error_profile)
        
        return processed_profiles
    
    def get_processing_stats(self, profiles: List[Dict]) -> Dict:
        """Get statistics about processed profiles"""
        if not profiles:
            return {}
        
        stats = {
            'total_profiles': len(profiles),
            'brazil_profiles': sum(1 for p in profiles if p.get('brazil_score', 0) > 0.5),
            'complete_profiles': sum(1 for p in profiles if p.get('completeness_score', 0) > 0.8),
            'high_quality_profiles': sum(1 for p in profiles if p.get('quality_score', 0) > 0.7),
            'experience_levels': {},
            'tech_categories': {},
            'processing_errors': sum(1 for p in profiles if 'processing_error' in p)
        }
        
        # Experience level distribution
        for profile in profiles:
            level = profile.get('experience_level', 'unknown')
            stats['experience_levels'][level] = stats['experience_levels'].get(level, 0) + 1
        
        # Tech category distribution
        for profile in profiles:
            tech_scores = profile.get('tech_category_scores', {})
            for category, score in tech_scores.items():
                if score > 0.1:  # Only count if there's some relevance
                    stats['tech_categories'][category] = stats['tech_categories'].get(category, 0) + 1
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    processor = ProfileProcessor()
    
    # Test profile
    test_profile = {
        'id': 'test_001',
        'name': 'João Silva Santos',
        'title': 'Senior Python Developer & Blockchain Specialist',
        'company': 'Nubank Ltda',
        'location': 'São Paulo, SP, Brasil',
        'experience': '5-8 years',
        'skills': ['Python', 'Django', 'AWS', 'Docker', 'Blockchain', 'Ethereum'],
        'certifications': ['AWS Solutions Architect', 'Certified Scrum Master'],
        'link': 'https://linkedin.com/in/joao-silva'
    }
    
    # Process profile
    processed = processor.clean_profile_data(test_profile)
    enhanced = processor.enhance_profile_for_scoring(processed, "Python blockchain developer")
    
    print("Original Profile:")
    for key, value in test_profile.items():
        print(f"  {key}: {value}")
    
    print("\nProcessed Profile (key additions):")
    print(f"  brazil_score: {enhanced['brazil_score']:.2f}")
    print(f"  quality_score: {enhanced['quality_score']:.2f}")
    print(f"  completeness_score: {enhanced['completeness_score']:.2f}")
    print(f"  experience_level: {enhanced['experience_level']}")
    print(f"  tech_category_scores: {enhanced['tech_category_scores']}")
    print(f"  keyword_features: {enhanced['keyword_features']}")
