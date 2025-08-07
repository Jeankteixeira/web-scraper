#!/usr/bin/env python3
"""
Test script for the LinkedIn Profile Scoring System
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from scraper import scrape_linkedin_profiles, test_scoring_system
    from scoring_engine import LinkedInProfileScorer
    from profile_processor import ProfileProcessor
    
    print("✅ All modules imported successfully!")
    
    def test_basic_scoring():
        """Test basic scoring functionality"""
        print("\n" + "="*60)
        print("🧪 TESTING BASIC SCORING FUNCTIONALITY")
        print("="*60)
        
        # Test with Python keyword
        print("\n1. Testing with 'Python' keyword...")
        profiles = scrape_linkedin_profiles("Python", location="Brasil")
        
        if profiles:
            print(f"✅ Found {len(profiles)} profiles")
            
            # Check if scoring data is present
            scored_profiles = [p for p in profiles if 'score' in p]
            print(f"✅ {len(scored_profiles)} profiles have scoring data")
            
            if scored_profiles:
                print("\n📊 TOP 3 SCORED PROFILES:")
                print("-" * 40)
                
                for i, profile in enumerate(scored_profiles[:3], 1):
                    print(f"\n{i}. {profile['name']}")
                    print(f"   Title: {profile['title']}")
                    print(f"   Company: {profile['company']}")
                    print(f"   Score: {profile.get('score', 0):.3f} ({profile.get('score_percentage', 0):.1f}%)")
                    print(f"   Classification: {profile.get('classification', 'N/A')}")
                    
                    explanations = profile.get('feature_explanations', [])
                    if explanations:
                        print(f"   Key factors:")
                        for exp in explanations[:3]:
                            print(f"     • {exp}")
                
                return True
            else:
                print("❌ No profiles have scoring data")
                return False
        else:
            print("❌ No profiles found")
            return False
    
    def test_blockchain_scoring():
        """Test scoring with blockchain-specific keywords"""
        print("\n" + "="*60)
        print("🔗 TESTING BLOCKCHAIN KEYWORD SCORING")
        print("="*60)
        
        print("\n2. Testing with 'blockchain hyperledger' keywords...")
        profiles = scrape_linkedin_profiles("blockchain hyperledger", location="Brasil")
        
        if profiles:
            scored_profiles = [p for p in profiles if 'score' in p]
            print(f"✅ Found {len(profiles)} profiles, {len(scored_profiles)} with scores")
            
            if scored_profiles:
                # Sort by score
                scored_profiles.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                print(f"\n🏆 HIGHEST SCORING PROFILE:")
                top_profile = scored_profiles[0]
                print(f"   Name: {top_profile['name']}")
                print(f"   Score: {top_profile.get('score', 0):.3f} ({top_profile.get('score_percentage', 0):.1f}%)")
                print(f"   Classification: {top_profile.get('classification', 'N/A')}")
                
                # Show feature breakdown
                features = top_profile.get('features', {})
                if features:
                    print(f"\n📈 FEATURE BREAKDOWN:")
                    for feature, value in features.items():
                        print(f"     {feature}: {value:.3f}")
                
                return True
            else:
                print("❌ No scored profiles found")
                return False
        else:
            print("❌ No profiles found")
            return False
    
    def test_scoring_components():
        """Test individual scoring components"""
        print("\n" + "="*60)
        print("🔧 TESTING SCORING COMPONENTS")
        print("="*60)
        
        try:
            # Test scorer initialization
            scorer = LinkedInProfileScorer()
            processor = ProfileProcessor()
            
            print("✅ Scorer and Processor initialized successfully")
            
            # Test with sample profile
            sample_profile = {
                'id': 'test_001',
                'name': 'Test User',
                'title': 'Senior Python Developer',
                'company': 'Nubank',
                'location': 'São Paulo, Brasil',
                'experience': '5-10',
                'skills': ['Python', 'Django', 'AWS'],
                'certifications': ['AWS Solutions Architect'],
                'languages': ['português', 'inglês'],
                'education': 'bacharel'
            }
            
            # Process profile
            processed = processor.clean_profile_data(sample_profile)
            enhanced = processor.enhance_profile_for_scoring(processed, "Python developer")
            
            print("✅ Profile processing successful")
            
            # Score profile
            result = scorer.calculate_score(enhanced, "Python developer")
            
            print("✅ Profile scoring successful")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Classification: {result['classification']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in scoring components: {e}")
            return False
    
    def main():
        """Run all tests"""
        print("🚀 LINKEDIN PROFILE SCORING SYSTEM TEST")
        print("="*60)
        
        tests_passed = 0
        total_tests = 3
        
        # Run tests
        if test_basic_scoring():
            tests_passed += 1
        
        if test_blockchain_scoring():
            tests_passed += 1
            
        if test_scoring_components():
            tests_passed += 1
        
        # Summary
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        print(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("🎉 ALL TESTS PASSED! Scoring system is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return tests_passed == total_tests

    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n💡 Make sure all required modules are available:")
    print("   - scoring_engine.py")
    print("   - profile_processor.py") 
    print("   - scraper.py")
    print("\n📦 You may need to install dependencies:")
    print("   pip install scikit-learn pandas numpy")
    sys.exit(1)

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
