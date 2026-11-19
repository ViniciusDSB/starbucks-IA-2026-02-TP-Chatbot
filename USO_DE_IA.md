# Declaração de uso de Inteligência Artificial generativa

## 1. Ferramentas usadas

| Ferramenta (nome e versão/modelo) | Como acessaram | Quem usou |
|---|---|---|
| Claude (Sonnet 5.5) | Navegador | Integrantes responsáveis pelas etapas correspondentes |
| ChatGPT (GPT-6) | Navegador | Integrantes responsáveis pelas etapas correspondentes |
| LLM do chatbot (gpt-6-luna) | OpenRouter, via código do projeto | Integrantes responsáveis pela implementação e testes |

## 2. Usos

| # | Etapa | O que a IA fez | Prompt (ou resumo + link para o registro) | O que o grupo fez com o resultado |
|---|---|---|---|---|
| 1 | Coleta de documentos | Gerou scripts para percorrer links, baixar PDFs e registrar as origens dos documentos em JSON. | Prompts para automatizar a coleta de documentos e o registro de suas fontes. | Inspecionou amostras dos documentos coletados e removeu aqueles que não eram relacionados ao escopo do projeto. |
| 2 | Organização e filtragem dos documentos | Organizou os documentos coletados e implementou filtragem simples baseada em termos, iterada sobre os dados. | Prompts para organização dos documentos e filtragem por termos relevantes. | Analisou as anomalias encontradas e escolheu os novos termos utilizados nas iterações de filtragem. |
| 3 | Registro das fontes | Preencheu o arquivo `FONTES.md` com base nos dados registrados em JSON. | Prompt para gerar o arquivo de fontes a partir do JSON editado após a filtragem. | Forneceu o JSON resultante da filtragem e verificou o registro das fontes. |
| 4 | Dados de treinamento | Claude e ChatGPT geraram frases para cada intenção, utilizando as intenções definidas em `grupo.yaml` e documentos de exemplo como contexto, respeitando a quantidade permitida. | Prompts para geração de frases por intenção, com as intenções e documentos de exemplo como contexto. | Realizou múltiplas solicitações, descartou frases repetidas e revisou os resultados. As frases geradas por IA não foram utilizadas no conjunto de teste. |
| 5 | Contagem dos dados | Gerou um script para contabilizar os dados de treinamento e teste. | Prompt para automatizar a contagem e verificar a distribuição e quantidade dos dados. | Utilizou as contagens para verificar se os conjuntos estavam de acordo com os requisitos do trabalho. |
| 6 | Filtragem com classificador | Gerou código para uma tentativa de filtragem de documentos utilizando o modelo MiniLM, incluindo as etapas de `filter` e `back filter`. | Prompts para implementar a filtragem de documentos utilizando o modelo MiniLM. | Abandonou a tentativa após avaliar os resultados e restaurou documentos úteis que haviam sido removidos. |
| 7 | Avaliação do limiar de confiança | Gerou scripts para verificar as probabilidades obtidas na validação cruzada, incluindo trechos de código utilizados no notebook. | Prompts para analisar as probabilidades da validação cruzada e avaliar diferentes limiares de confiança. | Descartou um trecho que produzia gráficos sem significado para a análise e utilizou a tabela de limiares para escolher o valor adotado. |
| 8 | Chatbot | O LLM, acessado via OpenRouter, redige respostas a partir dos trechos recuperados e da persona definida em `grupo.yaml`. | Prompts implementados no chatbot e configuração da persona em `grupo.yaml`. | Definiu a lógica de encaminhamento das consultas por meio dos classificadores. O LLM é utilizado apenas para redigir as respostas, não para definir a classificação ou a rota da consulta. |
| 9 | Revisão e formatação do relatório | ChatGPT gerou formatações para referências e tabelas a partir de anotações, além de tabelas em LaTeX com base nos dados do notebook. Claude revisou a gramática, a digitação e a pontuação do texto. | Prompts de formatação de referências e tabelas, geração de tabelas LaTeX com base nos dados e revisão textual do relatório. | Leu o relatório integralmente, revisou as alterações, verificou a coesão e procurou anomalias. Editou trechos em que a IA reescreveu partes do texto, buscando preservar a coesão e o conteúdo pretendido. Corrigiu erros de compilação LaTeX causados por sugestões do ChatGPT. |

## 3. O que NÃO foi feito com IA

A escrita das frases humanas utilizadas nos conjuntos de treinamento e teste, a escolha do limiar de confiança com base nos resultados obtidos e a escolha dos modelos de classificação de escopo e de intenção foram realizadas pelo grupo, sem utilização de IA generativa.

## 4. Reflexão

A IA acelerou a coleta e a organização dos documentos e a geração das frases de treino. Também errou ou limitou: a filtragem por termos removeu documentos úteis, a tentativa de filtragem com classificador foi abandonada, um trecho de código gerou um gráfico sem significado e as frases geradas vieram com muitas repetições. O grupo verificou os resultados inspecionando amostras de arquivos e contando os dados, e manteve para si as decisões de modelagem.