"""
Utility functions for the Web Scraper project
Consolidates common functionality and reduces code duplication
"""

import re
import logging
import requests
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import urllib.parse
from config import HTTP_CONFIG, VALIDATION_RULES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_keywords(keywords: str) -> str:
    """
    Validate and clean keywords input
    
    Args:
        keywords: Raw keywords string
        
    Returns:
        Cleaned keywords string
        
    Raises:
        ValidationError: If keywords are invalid
    """
    if not keywords or not keywords.strip():
        raise ValidationError("O campo 'palavras-chave' é obrigatório para realizar a busca.")
    
    cleaned_keywords = keywords.strip()
    
    if len(cleaned_keywords) < VALIDATION_RULES['keywords_min_length']:
        raise ValidationError("As palavras-chave devem ter pelo menos 2 caracteres.")
    
    return cleaned_keywords


def clean_text(text: Union[str, Any]) -> str:
    """
    Clean and normalize text fields
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep accents for Portuguese
    text = re.sub(r'[^\w\s\-\.\,\(\)\/\&\+]', '', text)
    
    return text


def standardize_list_field(field_value: Union[str, List[str], None], delimiter: str = ',') -> List[str]:
    """
    Standardize list fields (skills, certifications, etc.)
    
    Args:
        field_value: Input field value
        delimiter: Delimiter for string splitting
        
    Returns:
        Standardized list of strings
    """
    if not field_value:
        return []
    
    if isinstance(field_value, str):
        # Split by delimiter and clean
        items = re.split(f'[{delimiter};|]', field_value)
    elif isinstance(field_value, list):
        items = field_value
    else:
        return []
    
    standardized_items = []
    for item in items:
        if isinstance(item, str):
            item = item.strip().lower()
            if item and len(item) > 1:  # Filter out single characters
                standardized_items.append(item)
    
    return list(set(standardized_items))  # Remove duplicates


def extract_years_from_experience(experience: Union[str, int, None]) -> Dict[str, Optional[int]]:
    """
    Extract years of experience from various formats
    
    Args:
        experience: Experience field value
        
    Returns:
        Dictionary with min, max, and average years
    """
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


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate basic text similarity using word overlap
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def safe_http_request(url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
    """
    Make HTTP request with error handling and retries
    
    Args:
        url: Request URL
        method: HTTP method
        **kwargs: Additional request parameters
        
    Returns:
        Response object or None if failed
    """
    headers = kwargs.pop('headers', HTTP_CONFIG['headers'])
    timeout = kwargs.pop('timeout', HTTP_CONFIG['timeout'])
    max_retries = kwargs.pop('max_retries', HTTP_CONFIG['max_retries'])
    
    for attempt in range(max_retries):
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout, **kwargs)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, timeout=timeout, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"HTTP request attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All HTTP request attempts failed for URL: {url}")
                return None
    
    return None


def encode_url_params(params: Dict[str, str]) -> str:
    """
    Safely encode URL parameters
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        URL-encoded parameter string
    """
    return urllib.parse.urlencode(params, quote_plus=True)


def filter_by_year_range(items: List[Dict], year_field: str, year_range: str) -> List[Dict]:
    """
    Filter items by publication/experience year range
    
    Args:
        items: List of items to filter
        year_field: Field name containing year information
        year_range: Year range string (e.g., "2015-2025")
        
    Returns:
        Filtered list of items
    """
    if not year_range or '-' not in year_range:
        return items
    
    try:
        start_year, end_year = map(int, year_range.split('-'))
    except ValueError:
        logger.warning(f"Invalid year range format: {year_range}")
        return items
    
    filtered_items = []
    for item in items:
        year_value = item.get(year_field)
        if not year_value:
            continue
        
        # Extract year from various formats
        year_str = str(year_value)
        year_match = re.search(r'\d{4}', year_str)
        
        if year_match:
            try:
                item_year = int(year_match.group())
                if start_year <= item_year <= end_year:
                    filtered_items.append(item)
            except ValueError:
                continue
    
    return filtered_items


def normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to a specific range
    
    Args:
        value: Value to normalize
        min_val: Minimum value of range
        max_val: Maximum value of range
        
    Returns:
        Normalized value
    """
    return max(min_val, min(value, max_val))


def apply_sigmoid(x: float) -> float:
    """
    Apply sigmoid function for probability calculation
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid output between 0 and 1
    """
    import math
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def batch_process_with_progress(items: List[Any], process_func, batch_size: int = 50, 
                               description: str = "Processing") -> List[Any]:
    """
    Process items in batches with progress tracking
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        batch_size: Size of each batch
        description: Description for progress tracking
        
    Returns:
        List of processed items
    """
    processed_items = []
    total_items = len(items)
    
    logger.info(f"{description}: Starting processing of {total_items} items")
    
    for i in range(0, total_items, batch_size):
        batch = items[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_items + batch_size - 1) // batch_size
        
        logger.info(f"{description}: Processing batch {batch_num}/{total_batches}")
        
        for item in batch:
            try:
                processed_item = process_func(item)
                processed_items.append(processed_item)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                processed_items.append(item)  # Keep original on error
    
    logger.info(f"{description}: Completed processing {len(processed_items)} items")
    return processed_items


def create_timestamp() -> str:
    """
    Create ISO format timestamp string
    
    Returns:
        ISO format timestamp
    """
    return datetime.now().isoformat()


def safe_get_nested_value(data: Dict, keys: List[str], default: Any = None) -> Any:
    """
    Safely get nested dictionary value
    
    Args:
        data: Dictionary to search
        keys: List of nested keys
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def merge_dictionaries(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries, with later ones taking precedence
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def validate_profile_data(profile: Dict) -> bool:
    """
    Validate that profile contains minimum required data
    
    Args:
        profile: Profile dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['name', 'title']
    
    for field in required_fields:
        if not profile.get(field) or not str(profile[field]).strip():
            return False
    
    return True


def log_performance(func_name: str, start_time: datetime, item_count: int = 1):
    """
    Log performance metrics for a function
    
    Args:
        func_name: Name of the function
        start_time: Start time of execution
        item_count: Number of items processed
    """
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if item_count > 1:
        rate = item_count / duration if duration > 0 else 0
        logger.info(f"{func_name}: Processed {item_count} items in {duration:.2f}s ({rate:.1f} items/s)")
    else:
        logger.info(f"{func_name}: Completed in {duration:.2f}s")
