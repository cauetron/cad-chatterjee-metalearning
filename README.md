# CaD-Chatterjee: Meta-feature para Seleção de Algoritmos de Agrupamento

Projeto da disciplina de Meta-aprendizado, desenvolvido no contexto do Tema 5: **“Uma nova meta-feature para agrupamento”**. O trabalho parte do método CaD (*Correlation and Dissimilarity*) proposto por Pimentel e de Carvalho (2019) e avalia a substituição da correlação de Spearman por medidas alternativas de dependência, com foco no coeficiente de Chatterjee.

**Autores**

- Cauê Bittencourt Ângelo dos Santos — Instituto de Computação, Universidade Federal de Alagoas (UFAL)
- Davi Vieira Lourenço Correia — Instituto de Computação, Universidade Federal de Alagoas (UFAL)

## Resumo

O objetivo do projeto é investigar se medidas de correlação alternativas ao Spearman podem melhorar a caracterização de bases de dados para recomendação de algoritmos de agrupamento via meta-aprendizado.

Foram avaliados quatro conjuntos de meta-features:

- **Distance**: usa apenas dissimilaridade entre instâncias;
- **CaD-Sp**: combina dissimilaridade com correlação de Spearman;
- **CaD-Ke**: combina dissimilaridade com correlação de Kendall;
- **CaD-Ch**: combina dissimilaridade com o coeficiente de Chatterjee simetrizado pela média das duas direções.

O protocolo experimental utiliza 62 bases públicas da suíte OpenML CC-18, cinco algoritmos de agrupamento em nível base, dez índices internos para construção dos meta-alvos e dois meta-aprendizes: k-NN para ranking e Random Forest.

## Dados

Todos os dados utilizados são públicos e foram obtidos a partir da suíte **OpenML CC-18**.

Este repositório inclui apenas o arquivo `datasets/manifest.csv`, que registra as bases utilizadas no experimento. Os arquivos completos dos datasets não são versionados no repositório, mas podem ser regenerados pela etapa de aquisição e pré-processamento (`phase1_datasets.py`).

Após a filtragem, foram mantidas 62 bases. O processo de seleção considera bases com:

- pelo menos 100 instâncias;
- pelo menos 2 atributos numéricos;
- pelo menos 2 classes;
- atributos numéricos disponíveis após a remoção de atributos categóricos e valores ausentes.

Os datasets baixados a partir do OpenML mantêm suas licenças originais. A licença deste repositório cobre apenas o código, a documentação e os artefatos produzidos pelos autores.

## Estrutura do repositório

```text
.
├── phase1_datasets.py              # Aquisição e pré-processamento das bases OpenML
├── phase2_meta_targets.py          # Execução dos algoritmos de agrupamento e construção dos meta-alvos
├── phase3_meta_features.py         # Extração das meta-features Distance, CaD-Sp, CaD-Ke e CaD-Ch
├── phase4_meta_learning.py         # Avaliação dos meta-aprendizes e baselines
├── phase5_statistical_tests.py     # Testes estatísticos de Wilcoxon
├── phase6_figures_tables.py        # Geração de tabelas e figuras do relatório
├── requirements.txt                # Dependências Python
├── datasets/
│   └── manifest.csv                # Lista das bases usadas no experimento
├── meta_targets/                   # Rankings verdadeiros, índices internos e metadados da Fase 2
├── meta_features/                  # Meta-features extraídas
├── meta_learning/                  # Resultados completos, resumos e testes estatísticos
├── report_assets/                  # Tabelas e figuras geradas para o relatório
└── report/                         # Relatório final em PDF e/ou fontes LaTeX
```

Dependendo da versão enviada, algumas pastas de resultados intermediários podem já estar preenchidas. Caso contrário, elas são criadas durante a execução das fases.

### Arquivos derivados não versionados

A pasta `meta_targets/assignments/`, que contém os rótulos de cluster salvos em `.npy` para cálculo de ARI e AMI, **não é versionada** por conter muitos arquivos derivados. Ela é regenerada automaticamente pela `phase2_meta_targets.py`. Como esta pasta está ausente, a `phase4_meta_learning.py` só conseguirá recomputar ARI e AMI depois que a Fase 2 for executada. As métricas de similaridade entre rankings, como SRC, WRC, NDCG@3 e Top1Hit, não dependem diretamente desses arquivos de partições salvas.

## Instalação

Recomenda-se usar um ambiente virtual Python.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

Dependências principais:

- `openml`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`

## Como executar

O pipeline completo é dividido em seis fases.

### 1. Baixar e pré-processar os datasets

```bash
python phase1_datasets.py
```

Esta etapa baixa as bases da suíte OpenML CC-18, aplica os filtros definidos e gera os arquivos em `datasets/`, incluindo `datasets/manifest.csv`.

### 2. Construir os meta-alvos

```bash
python phase2_meta_targets.py
```

Esta fase executa os algoritmos de agrupamento em nível base e calcula os índices internos usados para construir o ranking verdadeiro de cada dataset. Ela também pode gerar a pasta `meta_targets/assignments/`, necessária para recomputar ARI e AMI na Fase 4.

Algoritmos avaliados:

- K-Means;
- Ward Agglomerative Clustering;
- Average Agglomerative Clustering;
- Gaussian Mixture com matriz diagonal;
- Gaussian Mixture com matriz cheia.

Os rankings verdadeiros são derivados de dez índices internos, agregados por ranking médio.

### 3. Extrair as meta-features

```bash
python phase3_meta_features.py
```

Esta etapa gera quatro conjuntos de meta-features:

- `mf_distance.csv`;
- `mf_cad_spearman.csv`;
- `mf_cad_kendall.csv`;
- `mf_cad_chatterjee.csv`.

Por padrão, a extração usa até 50.000 pares de instâncias por dataset.

### 4. Executar o meta-aprendizado

```bash
python phase4_meta_learning.py
```

Esta fase avalia os conjuntos de meta-features com:

- k-NN para ranking, com `k=5` e agregação por *average ranking*;
- Random Forest com 100 árvores;
- baselines MeanRank e MajorityRank.

As métricas principais são:

- SRC;
- WRC;
- NDCG@3;
- Top1Hit;
- ARI;
- AMI.

Observação: para recomputar ARI e AMI, a Fase 4 precisa das partições salvas em `meta_targets/assignments/`. Se essa pasta não estiver presente, execute antes a Fase 2 para regenerá-la.

### 5. Executar os testes estatísticos

```bash
python phase5_statistical_tests.py
```

Esta etapa aplica testes de Wilcoxon pareados, incluindo a análise agregada por dataset, usada como evidência estatística principal no relatório.

### 6. Gerar tabelas e figuras

```bash
python phase6_figures_tables.py
```

Esta fase gera tabelas e figuras utilizadas no relatório em `report_assets/`.

## Teste rápido

Para validar o funcionamento do pipeline sem executar todos os datasets, é possível usar um subconjunto reduzido:

```bash
python phase1_datasets.py --max-datasets 10
python phase2_meta_targets.py --max-datasets 5 --sort-by-size
python phase3_meta_features.py --max-datasets 5 --sort-by-size
python phase4_meta_learning.py --repetitions 2 --folds 5
python phase5_statistical_tests.py
python phase6_figures_tables.py
```

Esse teste serve apenas para verificar a execução do código. Os resultados reportados no trabalho foram obtidos com a configuração completa.

## Resultados principais

Os resultados indicam que a versão CaD-Chatterjee apresenta ganhos médios em métricas de similaridade entre rankings, especialmente com k-NN, quando comparada ao CaD-Spearman. No entanto, esses ganhos não foram estatisticamente significativos na análise agregada por dataset e não se traduziram de forma consistente em melhora nas métricas externas ARI e AMI.

Assim, o coeficiente de Chatterjee mostrou-se uma alternativa promissora para caracterização de dados em meta-aprendizado para agrupamento, mas sem evidência suficiente de superioridade robusta neste protocolo experimental.

## Reprodutibilidade

As principais etapas usam semente fixa, com `random_state=42` por padrão. Os arquivos intermediários e metadados de execução são salvos nas pastas `meta_targets/`, `meta_features/` e `meta_learning/`.

A pasta `meta_targets/assignments/` é derivada e pode ser regenerada pela Fase 2. Ela foi omitida do versionamento por conter muitos arquivos `.npy`.

Pequenas diferenças podem ocorrer em função de versão de bibliotecas, paralelização ou ambiente de execução.

## Relatório

O relatório final está disponível na pasta `report/`.

Os principais artefatos gerados pelo pipeline são:

- `meta_learning/results_summary.csv`: resumo das métricas por método;
- `meta_learning/statistical_tests_by_dataset.csv`: testes de Wilcoxon agregados por dataset;
- `report_assets/`: tabelas e figuras usadas no relatório.

## Licença

Este projeto é disponibilizado sob a licença MIT. Consulte o arquivo `LICENSE`.

Os datasets obtidos via OpenML não são relicenciados por este repositório e permanecem sujeitos às suas respectivas licenças originais.
