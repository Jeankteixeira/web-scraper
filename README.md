# Web Scraper

## Descrição
Este projeto realiza extração automatizada de dados de páginas web. Utiliza Python para o processamento principal e HTML para uma interface básica, tornando possível extrair, processar e pontuar perfis conforme critérios configuráveis.

## Funcionalidades
- Extração eficiente de dados web
- Processamento e análise de perfis
- Engine de pontuação customizável
- Interface visual básica em HTML
- Estrutura modular para fácil ampliação

## Estrutura do Projeto
- `scraper.py`: Script principal para execução do scraping web.
- `scoring_engine.py`: Módulo responsável por atribuir pontuações aos dados/usuários extraídos.
- `profile_processor.py`: Processa informações de perfis coletados.
- `config.py`: Arquivo de configurações do scraper (alvos, critérios etc.).
- `requirements.txt`: Lista de dependências do projeto.
- `utils.py`: Funções auxiliares comuns ao projeto.
- `web_scraper.html`: Interface visual básica para demonstrar a execução ou resultados.
- `test_scoring.py` e `test_scoring_system.py`: Scripts de testes automatizados.
- `.vscode`: Configurações para ambiente VSCode.

## Instalação
```bash
git clone https://github.com/Jeankteixeira/web-scraper.git
cd web-scraper
pip install -r requirements.txt
```

## Como Usar
Execute o script principal para realizar o scraping:
```bash
python scraper.py
```

Edite as configurações conforme necessário em `config.py`.

## Testes
Para rodar os testes automatizados de scoring:
```bash
python -m unittest test_scoring.py
python -m unittest test_scoring_system.py
```

## Contribuição
Contribuições são bem-vindas! Siga os passos abaixo:
1. Faça um fork do projeto
2. Crie uma branch para sua feature/fix
3. Submeta um Pull Request

## Autor
Jean Kalust
