"""
Configuration file for Web Scraper project
Centralizes constants, settings, and configuration data
"""

# Academic Search Configuration
ACADEMIC_CONFIG = {
    'free_sources': ['google-scholar', 'arxiv', 'scielo', 'dblp'],
    'source_urls': {
        'google-scholar': 'https://scholar.google.com/scholar?q={}',
        'arxiv': 'http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results=20',
        'scielo': 'https://search.scielo.org/?q={}&lang={}&count=20&from=0&output=site&sort=&format=summary&fb=&page=1',
        'dblp': 'https://dblp.org/search?q={}'
    },
    'language_keywords': {
        'portuguese': ['português', 'brazil', 'brasil', 'pt'],
        'english': ['english', 'en'],
        'spanish': ['español', 'spanish', 'es']
    },
    'area_keywords': {
        'computing': 'computer science programming software',
        'engineering': 'engineering technology',
        'health': 'medicine health medical'
    }
}

# LinkedIn Profile Configuration
LINKEDIN_CONFIG = {
    'relevant_companies': {
        'nubank', 'ifood', 'stone', 'mercado livre', 'mercadolibre', 'globo',
        'petrobras', 'vale', 'itau', 'bradesco', 'santander', 'banco do brasil',
        'ambev', 'jbs', 'embraer', 'natura', 'magazine luiza', 'via varejo',
        'b3', 'localiza', 'suzano', 'klabin', 'ultrapar', 'cosan',
        'google', 'microsoft', 'amazon', 'meta', 'apple', 'netflix',
        'uber', 'airbnb', 'spotify', 'linkedin', 'twitter', 'tesla'
    },
    'seniority_mapping': {
        'estagiario': 0, 'trainee': 0, 'intern': 0, 'internship': 0,
        'junior': 1, 'jr': 1, 'entry': 1, 'associate': 1,
        'pleno': 2, 'mid': 2, 'middle': 2, 'mid-level': 2,
        'senior': 3, 'sr': 3, 'sênior': 3, 'mid-senior': 3,
        'principal': 4, 'staff': 4, 'lead': 4, 'tech lead': 4,
        'manager': 4, 'gerente': 4, 'coordenador': 4,
        'director': 5, 'diretor': 5, 'head': 5, 'vp': 5,
        'cto': 6, 'ceo': 6, 'founder': 6, 'co-founder': 6,
        'president': 6, 'presidente': 6, 'executive': 5
    },
    'brazil_patterns': [
        r'\bbrasil\b', r'\bbrazil\b', r'\bbr\b',
        r'são paulo', r'rio de janeiro', r'belo horizonte',
        r'salvador', r'fortaleza', r'brasília', r'curitiba',
        r'recife', r'porto alegre', r'goiânia', r'belém',
        r'guarulhos', r'campinas', r'nova iguaçu', r'maceió',
        r'duque de caxias', r'natal', r'teresina', r'campo grande',
        r'joão pessoa', r'são luís', r'aracaju', r'florianópolis',
        r'vitória', r'macapá', r'rio branco', r'boa vista',
        r'palmas', r'cuiabá', r'porto velho'
    ]
}

# Technology Categories for Skills Matching
TECH_CATEGORIES = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#',
        'go', 'rust', 'kotlin', 'swift', 'php', 'ruby', 'scala',
        'r', 'matlab', 'sql', 'html', 'css'
    ],
    'frameworks': [
        'react', 'angular', 'vue', 'django', 'flask', 'spring',
        'express', 'node.js', 'laravel', 'rails', 'asp.net',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn'
    ],
    'databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
        'cassandra', 'dynamodb', 'oracle', 'sql server'
    ],
    'cloud_platforms': [
        'aws', 'azure', 'gcp', 'google cloud', 'heroku',
        'digitalocean', 'linode', 'alibaba cloud'
    ],
    'devops_tools': [
        'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
        'terraform', 'ansible', 'chef', 'puppet', 'vagrant'
    ],
    'blockchain': [
        'blockchain', 'ethereum', 'bitcoin', 'solidity',
        'hyperledger', 'smart contracts', 'web3', 'defi',
        'nft', 'cryptocurrency', 'consensus', 'mining'
    ]
}

# Scoring Engine Configuration
SCORING_CONFIG = {
    'feature_weights': {
        'keyword_similarity': 2.5,
        'seniority_score': 0.8,
        'years_experience': 0.3,
        'relevant_company': 1.0,
        'brazil_indicator': 1.2,
        'certifications_count': 0.5,
        'skills_match': 1.5
    },
    'bias': -5.0,
    'score_thresholds': {
        'high': 0.85,
        'medium': 0.6,
        'low': 0.0
    },
    'classifications': {
        'high': 'Alta relevância',
        'medium': 'Relevância moderada',
        'low': 'Baixa relevância'
    }
}

# HTTP Configuration
HTTP_CONFIG = {
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    },
    'timeout': 30,
    'max_retries': 3
}

# Validation Rules
VALIDATION_RULES = {
    'keywords_min_length': 2,
    'max_results_per_source': 50,
    'max_profiles_batch': 200
}

# File Export Configuration
EXPORT_CONFIG = {
    'supported_formats': ['csv', 'json', 'excel'],
    'csv_delimiter': ',',
    'json_indent': 2,
    'max_export_size': 10000
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_max_bytes': 10485760,  # 10MB
    'backup_count': 5
}
