import requests
+import urllib.parse
+from bs4 import BeautifulSoup
+import csv
+import json
+
+from scoring_engine import LinkedInProfileScorer
+from profile_processor import ProfileProcessor
+from utils import validate_keywords
+
+SCORING_AVAILABLE = True
+
+
+def search_academic_articles(keywords, sources=None, publication_year=None, authors=None,
+                             language=None, knowledge_area=None, affiliated_institution=None,
+                             author_country=None):
+    # Validate keywords
+    keywords = validate_keywords(keywords)
+
+    results = []
+    free_sources = ['google-scholar', 'arxiv', 'scielo', 'dblp']
+
+    if sources is None:
+        sources = free_sources
+    else:
+        sources = [source for source in sources if source in free_sources]
+
+    enhanced_keywords = build_enhanced_query(keywords, authors, affiliated_institution,
+                                            knowledge_area, language)
+
+    try:
+        if 'google-scholar' in sources:
+            results.extend(search_google_scholar(enhanced_keywords))
+        if 'arxiv' in sources:
+            results.extend(search_arxiv(enhanced_keywords))
+        if 'scielo' in sources:
+            results.extend(search_scielo(enhanced_keywords, language))
+        if 'dblp' in sources:
+            results.extend(search_dblp(enhanced_keywords))
+
+        results = apply_post_search_filters(results, publication_year, authors,
+                                            language, knowledge_area, affiliated_institution,
+                                            author_country)
+    except Exception as e:
+        print(f"Erro durante a busca: {e}")
+        return []
+
+    return results
+
+
+def build_enhanced_query(keywords, authors=None, institution=None, knowledge_area=None, language=None):
+    query_parts = [keywords]
+
+    if authors:
+        author_list = [author.strip() for author in authors.split(',') if author.strip()]
+        for author in author_list:
+            query_parts.append(f'author:"{author}"')
+
+    if institution:
+        query_parts.append(f'"{institution}"')
+
+    if knowledge_area:
+        area_keywords = {
+            'computing': 'computer science programming software',
+            'engineering': 'engineering technology',
+            'health': 'medicine health medical'
+        }
+        if knowledge_area in area_keywords:
+            query_parts.append(area_keywords[knowledge_area])
+
+    return ' '.join(query_parts)
+
+
+def apply_post_search_filters(results, publication_year=None, authors=None, language=None,
+                              knowledge_area=None, affiliated_institution=None, author_country=None):
+    filtered_results = results
+
+    if publication_year:
+        year_range = publication_year.split('-')
+        if len(year_range) == 2:
+            start_year, end_year = int(year_range[0]), int(year_range[1])
+            filtered_results = [r for r in filtered_results
+                                if filter_by_year(r, start_year, end_year)]
+
+    if language:
+        filtered_results = [r for r in filtered_results
+                            if filter_by_language(r, language)]
+
+    if affiliated_institution:
+        filtered_results = [r for r in filtered_results
+                            if filter_by_institution(r, affiliated_institution)]
+
+    return filtered_results
+
+
+def filter_by_year(result, start_year, end_year):
+    if 'published' in result:
+        try:
+            pub_year = int(result['published'][:4])
+            return start_year <= pub_year <= end_year
+        except (ValueError, IndexError):
+            pass
+    elif 'year' in result:
+        try:
+            pub_year = int(result['year'])
+            return start_year <= pub_year <= end_year
+        except (ValueError, TypeError):
+            pass
+    return True
+
+
+def filter_by_language(result, language):
+    language_keywords = {
+        'portuguese': ['português', 'brazil', 'brasil', 'pt'],
+        'english': ['english', 'en'],
+        'spanish': ['español', 'spanish', 'es']
+    }
+
+    if language in language_keywords:
+        keywords = language_keywords[language]
+        text_to_check = (result.get('title', '') + ' ' +
+                         result.get('authors', '') + ' ' +
+                         result.get('snippet', '')).lower()
+        return any(keyword in text_to_check for keyword in keywords)
+
+    return True
+
+
+def filter_by_institution(result, institution):
+    text_to_check = (result.get('authors', '') + ' ' +
+                     result.get('snippet', '')).lower()
+    return institution.lower() in text_to_check
+
+
+def search_google_scholar(keywords):
+    results = []
+    try:
+        encoded_keywords = urllib.parse.quote_plus(keywords)
+        google_scholar_url = f"https://scholar.google.com/scholar?q={encoded_keywords}"
+
+        headers = {
+            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
+        }
+
+        response = requests.get(google_scholar_url, headers=headers)
+        soup = BeautifulSoup(response.text, 'html.parser')
+
+        for item in soup.select('.gs_ri'):
+            title_element = item.select_one('.gs_rt')
+            link_element = item.select_one('.gs_rt a')
+            authors_element = item.select_one('.gs_a')
+            snippet_element = item.select_one('.gs_rs')
+
+            if title_element:
+                title = title_element.get_text().strip()
+                link = link_element['href'] if link_element and link_element.get('href') else ''
+                authors = authors_element.get_text().strip() if authors_element else ''
+                snippet = snippet_element.get_text().strip() if snippet_element else ''
+
+                results.append({
+                    'title': title,
+                    'link': link,
+                    'authors': authors,
+                    'snippet': snippet,
+                    'source': 'Google Scholar'
+                })
+    except Exception as e:
+        print(f"Erro ao buscar no Google Scholar: {e}")
+
+    return results
+
+
+def search_arxiv(keywords):
+    results = []
+    try:
+        encoded_keywords = urllib.parse.quote_plus(keywords)
+        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_keywords}&start=0&max_results=20"
+
+        response = requests.get(arxiv_url)
+        soup = BeautifulSoup(response.text, 'xml')
+
+        for entry in soup.find_all('entry'):
+            title_element = entry.find('title')
+            link_element = entry.find('id')
+            authors_elements = entry.find_all('author')
+            summary_element = entry.find('summary')
+            published_element = entry.find('published')
+
+            if title_element:
+                title = title_element.get_text().strip()
+                link = link_element.get_text().strip() if link_element else ''
+                authors = ', '.join([author.find('name').get_text() for author in authors_elements if author.find('name')])
+                snippet = summary_element.get_text().strip() if summary_element else ''
+                published = published_element.get_text().strip()[:10] if published_element else ''
+
+                results.append({
+                    'title': title,
+                    'link': link,
+                    'authors': authors,
+                    'snippet': snippet,
+                    'published': published,
+                    'source': 'arXiv'
+                })
+    except Exception as e:
+        print(f"Erro ao buscar no arXiv: {e}")
+
+    return results
+
+
+def search_scielo(keywords, language=None):
+    results = []
+    try:
+        encoded_keywords = urllib.parse.quote_plus(keywords)
+
+        lang_param = 'pt'
+        if language == 'english':
+            lang_param = 'en'
+        elif language == 'spanish':
+            lang_param = 'es'
+
+        scielo_url = f"https://search.scielo.org/?q={encoded_keywords}&lang={lang_param}&count=20&from=0&output=site&sort=&format=summary&fb=&page=1"
+
+        headers = {
+            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
+        }
+
+        response = requests.get(scielo_url, headers=headers)
+        soup = BeautifulSoup(response.text, 'html.parser')
+
+        for item in soup.select('.results .item'):
+            title_element = item.select_one('.title a')
+            authors_element = item.select_one('.authors')
+            snippet_element = item.select_one('.abstract')
+
+            if title_element:
+                title = title_element.get_text().strip()
+                link = title_element.get('href', '')
+                authors = authors_element.get_text().strip() if authors_element else ''
+                snippet = snippet_element.get_text().strip() if snippet_element else ''
+
+                results.append({
+                    'title': title,
+                    'link': link,
+                    'authors': authors,
+                    'snippet': snippet,
+                    'source': 'SciELO'
+                })
+    except Exception as e:
+        print(f"Erro ao buscar no SciELO: {e}")
+
+    return results
+
+
+def search_dblp(keywords):
+    results = []
+    try:
+        encoded_keywords = urllib.parse.quote_plus(keywords)
+        dblp_url = f"https://dblp.org/search?q={encoded_keywords}"
+
+        headers = {
+            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
+        }
+
+        response = requests.get(dblp_url, headers=headers)
+        soup = BeautifulSoup(response.text, 'html.parser')
+
+        for item in soup.select('.publ'):
+            title_element = item.select_one('.title')
+            authors_element = item.select_one('.authors')
+            venue_element = item.select_one('.venue')
+            year_element = item.select_one('.year')
+
+            if title_element:
+                title = title_element.get_text().strip()
+                link = title_element.find('a')['href'] if title_element.find('a') else ''
+                authors = authors_element.get_text().strip() if authors_element else ''
+                venue = venue_element.get_text().strip() if venue_element else ''
+                year = year_element.get_text().strip() if year_element else ''
+
+                results.append({
+                    'title': title,
+                    'link': link,
+                    'authors': authors,
+                    'venue': venue,
+                    'year': year,
+                    'source': 'DBLP'
+                })
+    except Exception as e:
+        print(f"Erro ao buscar no DBLP: {e}")
+
+    return results
+
+
+def scrape_linkedin_profiles(keywords, location=None, job_seniority=None, experience_ranges=None,
+                             industries=None, technical_skills=None, languages=None,
+                             company=None, education=None, certifications=None, enable_scoring=True):
+    mock_profiles = generate_mock_linkedin_profiles(keywords, location, job_seniority,
+                                                   experience_ranges, industries,
+                                                   technical_skills, company)
+
+    if enable_scoring and SCORING_AVAILABLE and keywords:
+        try:
+            processor = ProfileProcessor()
+            scorer = LinkedInProfileScorer()
+
+            processed_profiles = processor.batch_process_profiles(mock_profiles, keywords)
+            scored_profiles = scorer.batch_score_profiles(processed_profiles, keywords)
+
+            print(f"Successfully scored {len(scored_profiles)} profiles")
+            return scored_profiles
+        except Exception as e:
+            print(f"Error in scoring pipeline: {e}")
+            print("Returning profiles without scoring...")
+            return mock_profiles
+
+    return mock_profiles
+
+
+def generate_mock_linkedin_profiles(keywords, location, job_seniority, experience_ranges,
+                                   industries, technical_skills, company):
+    base_profiles = [
+        {
+            'id': 'profile_001',
+            'name': 'Carlos Silva',
+            'title': 'Senior Python Developer & Blockchain Specialist',
+            'company': 'Nubank',
+            'location': 'São Paulo, SP, Brasil',
+            'experience': '5-10',
+            'seniority': 'senior',
+            'industry': 'fintech',
+            'skills': ['Python', 'Django', 'AWS', 'Docker', 'Blockchain', 'Ethereum', 'Hyperledger Besu'],
+            'languages': ['português', 'inglês'],
+            'education': 'bacharel',
+            'certifications': ['AWS Solutions Architect', 'Certified Ethereum Developer'],
+            'link': 'https://linkedin.com/in/carlos-silva-dev',
+            'summary': 'Desenvolvedor Python sênior com 7 anos de experiência em fintech e blockchain. Especialista em Hyperledger Besu e smart contracts.'
+        },
+        {
+            'id': 'profile_002',
+            'name': 'Ana Oliveira',
+            'title': 'Full Stack Developer - Blockchain & Web3',
+            'company': 'iFood',
+            'location': 'São Paulo, SP, Brasil',
+            'experience': '2-5',
+            'seniority': 'pleno',
+            'industry': 'tecnologia',
+            'skills': ['JavaScript', 'React', 'Node.js', 'MongoDB', 'Solidity', 'Web3', 'Smart Contracts'],
+            'languages': ['português', 'inglês'],
+            'education': 'bacharel',
+            'certifications': ['Scrum Master', 'Blockchain Developer Certification'],
+            'link': 'https://linkedin.com/in/ana-oliveira-fullstack',
+            'summary': 'Desenvolvedora full stack com foco em tecnologias blockchain e Web3. Experiência em DeFi e NFTs.'
+        },
+        {
+            'id': 'profile_003',
+            'name': 'Roberto Santos',
+            'title': 'Junior Software Engineer',
+            'company': 'Stone',
+            'location': 'Rio de Janeiro, RJ, Brasil',
+            'experience': '0-2',
+            'seniority': 'junior',
+            'industry': 'fintech',
+            'skills': ['Java', 'Spring Boot', 'MySQL', 'Git', 'Docker'],
+            'languages': ['português'],
+            'education': 'bacharel',
+            'certifications': [],
+            'link': 'https://linkedin.com/in/roberto-santos-jr',
+            'summary': 'Engenheiro de software júnior apaixonado por tecnologia e inovação financeira.'
+        },
+        {
+            'id': 'profile_004',
+            'name': 'Mariana Costa',
+            'title': 'Data Scientist & ML Engineer',
+            'company': 'Mercado Livre',
+            'location': 'Buenos Aires, Argentina',
+            'experience': '2-5',
+            'seniority': 'pleno',
+            'industry': 'ecommerce',
+            'skills': ['Python', 'Machine Learning', 'TensorFlow', 'SQL', 'Apache Spark', 'Kubernetes'],
+            'languages': ['português', 'espanhol', 'inglês'],
+            'education': 'mestrado',
+            'certifications': ['Google Cloud Professional ML Engineer', 'AWS Machine Learning'],
+            'link': 'https://linkedin.com/in/mariana-costa-ds',
+            'summary': 'Cientista de dados com mestrado em IA, especializada em sistemas de recomendação e processamento de big data.'
+        },
+        {
+            'id': 'profile_005',
+            'name': 'João Pereira',
+            'title': 'DevOps Engineer & Cloud Architect',
+            'company': 'Globo',
+            'location': 'Rio de Janeiro, RJ, Brasil',
+            'experience': '5-10',
+            'seniority': 'senior',
+            'industry': 'tecnologia',
+            'skills': ['Docker', 'Kubernetes', 'AWS', 'Terraform', 'Jenkins', 'Python', 'Monitoring'],
+            'languages': ['português', 'inglês'],
+            'education': 'bacharel',
+            'certifications': ['AWS DevOps Professional', 'Kubernetes Administrator', 'Terraform Associate'],
+            'link': 'https://linkedin.com/in/joao-pereira-devops',
+            'summary': 'Engenheiro DevOps sênior com 8 anos de experiência em cloud computing e automação de infraestrutura.'
+        },
+        {
+            'id': 'profile_006',
+            'name': 'Fernanda Lima',
+            'title': 'Senior Product Manager - Fintech',
+            'company': 'Uber',
+            'location': 'São Paulo, SP, Brasil',
+            'experience': '5-10',
+            'seniority': 'senior',
+            'industry': 'tecnologia',
+            'skills': ['Product Strategy', 'Agile', 'Analytics', 'SQL', 'A/B Testing', 'User Research'],
+            'languages': ['português', 'inglês'],
+            'education': 'mba',
+            'certifications': ['PMP', 'Scrum Product Owner', 'Google Analytics'],
+            'link': 'https://linkedin.com/in/fernanda-lima-pm',
+            'summary': 'Gerente de produto sênior com MBA e 7 anos de experiência em produtos digitais e fintech.'
+        },
+        {
+            'id': 'profile_007',
+            'name': 'Lucas Rodrigues',
+            'title': 'Blockchain Developer & Smart Contract Auditor',
+            'company': 'Chainlink Labs',
+            'location': 'São Paulo, SP, Brasil',
+            'experience': '3-5',
+            'seniority': 'pleno',
+            'industry': 'blockchain',
+            'skills': ['Solidity', 'Ethereum', 'Hyperledger Besu', 'Smart Contracts', 'DeFi', 'Security Auditing'],
+            'languages': ['português', 'inglês'],
+            'education': 'bacharel',
+            'certifications': ['Certified Blockchain Security Professional', 'Ethereum Developer'],
+            'link': 'https://linkedin.com/in/lucas-rodrigues-blockchain',
+            'summary': 'Desenvolvedor blockchain especializado em Hyperledger Besu e auditoria de smart contracts. Contribuidor ativo em projetos DeFi.'
+        },
+        {
+            'id': 'profile_008',
+            'name': 'Patricia Mendes',
+            'title': 'CTO & Blockchain Architect',
+            'company': 'Foxbit',
+            'location': 'São Paulo, SP, Brasil',
+            'experience': '10+',
+            'seniority': 'executive',
+            'industry': 'cryptocurrency',
+            'skills': ['Blockchain Architecture', 'Hyperledger', 'Ethereum', 'Team Leadership', 'System Design'],
+            'languages': ['português', 'inglês', 'espanhol'],
+            'education': 'mestrado',
+            'certifications': ['Certified Blockchain Architect', 'AWS Solutions Architect', 'PMP'],
+            'link': 'https://linkedin.com/in/patricia-mendes-cto',
+            'summary': 'CTO com 12 anos de experiência em tecnologia, especializada em arquitetura blockchain e liderança técnica.'
+        }
+    ]
+
+    filtered_profiles = base_profiles
+
+    if keywords:
+        keywords_lower = keywords.lower()
+        filtered_profiles = [p for p in filtered_profiles
+                             if keywords_lower in p['title'].lower() or
+                             keywords_lower in p['name'].lower() or
+                             any(keywords_lower in skill.lower() for skill in p['skills'])]
+
+    if location:
+        location_lower = location.lower()
+        filtered_profiles = [p for p in filtered_profiles
+                             if location_lower in p['location'].lower()]
+
+    if job_seniority:
+        filtered_profiles = [p for p in filtered_profiles
+                             if p['seniority'] == job_seniority]
+
+    if experience_ranges:
+        filtered_profiles = [p for p in filtered_profiles
+                             if p['experience'] in experience_ranges]
+
+    if industries:
+        filtered_profiles = [p for p in filtered_profiles
+                             if p['industry'] in industries]
+
+    if technical_skills:
+        filtered_profiles = [p for p in filtered_profiles
+                             if any(skill in p['skills'] for skill in technical_skills)]
+
+    if company:
+        company_lower = company.lower()
+        filtered_profiles = [p for p in filtered_profiles
+                             if company_lower in p['company'].lower()]
+
+    result_profiles = []
+    for profile in filtered_profiles:
+        result_profiles.append({
+            'id': profile['id'],
+            'name': profile['name'],
+            'title': profile['title'],
+            'company': profile['company'],
+            'location': profile['location'],
+            'experience': profile['experience'],
+            'skills': profile['skills'],
+            'languages': profile['languages'],
+            'education': profile['education'],
+            'certifications': profile['certifications'],
+            'link': profile['link'],
+            'summary': profile.get('summary', ''),
+            'seniority': profile['seniority'],
+            'industry': profile['industry']
+        })
+
+    return result_profiles
+
+
+def export_results(results, filename, filetype='csv'):
+    if filetype == 'csv':
+        with open(filename, mode='w', newline='') as file:
+            writer = csv.DictWriter(file, fieldnames=results[0].keys())
+            writer.writeheader()
+            writer.writerows(results)
+    elif filetype == 'json':
+        with open(filename, 'w') as file:
+            json.dump(results, file, indent=4)
+
+
+def test_scoring_system():
+    if not SCORING_AVAILABLE:
+        print("Sistema de scoring não disponível. Instale as dependências necessárias.")
+        return
+
+    print("=== Teste do Sistema de Scoring ===")
+
+    test_keywords = "blockchain hyperledger besu"
+    profiles = generate_mock_linkedin_profiles(test_keywords, "Brasil", None, None, None, None, None)
+
+    processor = ProfileProcessor()
+    scorer = LinkedInProfileScorer()
+
+    processed_profiles = processor.batch_process_profiles(profiles, test_keywords)
+    scored_profiles = scorer.batch_score_profiles(processed_profiles, test_keywords)
+
+    print(f"\nResultados para palavras-chave: '{test_keywords}'")
+    print("=" * 60)
+
+    for profile in scored_profiles[:5]:
+        print(f"\n{profile['name']} - {profile['title']}")
+        print(f"Empresa: {profile['company']} | Local: {profile['location']}")
+        print(f"Score: {profile['score']:.3f} ({profile['score_percentage']}%) - {profile['classification']}")
+        print("Fatores de relevância:")
+        for explanation in profile['feature_explanations']:
+            print(f"  • {explanation}")
+        print("-" * 40)
+
+    stats = processor.get_processing_stats(processed_profiles)
+    print(f"\n=== Estatísticas ===")
+    print(f"Total de perfis: {stats['total_profiles']}")
+    print(f"Perfis brasileiros: {stats['brazil_profiles']}")
+    print(f"Perfis completos: {stats['complete_profiles']}")
+    print(f"Perfis de alta qualidade: {stats['high_quality_profiles']}")
+
+
+def main():
+    print("Bem-vindo ao sistema de web scraping com scoring de relevância!")
+    print("1. Busca de artigos acadêmicos")
+    print("2. Busca de perfis LinkedIn com scoring")
+    print("3. Teste do sistema de scoring")
+
+    choice = input("\nEscolha uma opção (1-3): ").strip()
+
+    if choice == "1":
+        while True:
+            keywords = input("Digite as palavras-chave para artigos acadêmicos: ")
+            try:
+                articles = search_academic_articles(keywords)
+                print(f"Artigos encontrados: {len(articles)} resultados")
+                for i, article in enumerate(articles[:5], 1):
+                    print(f"{i}. {article['title']} - {article['source']}")
+                break
+            except ValueError as e:
+                print(e)
+                print("Por favor, tente novamente.\n")
+
+    elif choice == "2":
+        linkedin_keywords = input("Digite as palavras-chave para perfis do LinkedIn: ")
+        if linkedin_keywords.strip():
+            location = input("Digite a localização (opcional, padrão: Brasil): ") or "Brasil"
+
+            try:
+                profiles = scrape_linkedin_profiles(linkedin_keywords, location, enable_scoring=True)
+                print(f"\nPerfis encontrados e pontuados: {len(profiles)} resultados")
+
+                if SCORING_AVAILABLE:
+                    print("\n=== Top 3 Perfis por Relevância ===")
+                    for i, profile in enumerate(profiles[:3], 1):
+                        print(f"\n{i}. {profile['name']} - {profile['title']}")
+                        print(f"   Empresa: {profile['company']}")
+                        if 'score' in profile:
+                            print(f"   Score: {profile['score']:.3f} ({profile['score_percentage']}%) - {profile['classification']}")
+                            if profile.get('feature_explanations'):
+                                print("   Fatores: " + ", ".join(profile['feature_explanations'][:2]))
+                else:
+                    for i, profile in enumerate(profiles[:3], 1):
+                        print(f"{i}. {profile['name']} - {profile['title']} ({profile['company']})")
+
+            except Exception as e:
+                print(f"Erro ao buscar perfis do LinkedIn: {e}")
+                profiles = []
+        else:
+            print("Palavras-chave são obrigatórias para busca do LinkedIn.")
+            profiles = []
+
+    elif choice == "3":
+        test_scoring_system()
+        return
+
+    else:
+        print("Opção inválida.")
+        return
+
+    if 'articles' in locals() and articles:
+        results_to_export = articles
+    elif 'profiles' in locals() and profiles:
+        results_to_export = profiles
+    else:
+        results_to_export = []
+
+    if results_to_export:
+        export_choice = input("\nDeseja exportar os resultados? (csv/json/n para não): ").lower()
+        if export_choice in ['csv', 'json']:
+            filename = input("Digite o nome do arquivo: ")
+            try:
+                export_results(results_to_export, f"{filename}.{export_choice}", export_choice)
+                print(f"Resultados exportados para {filename}.{export_choice}")
+            except Exception as e:
+                print(f"Erro ao exportar resultados: {e}")
+    else:
+        print("Nenhum resultado encontrado para exportar.")
+
+
+if __name__ == "__main__":
        