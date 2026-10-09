# Trabalho Prático de IA (2026/2): chatbot com classificação de intenções e RAG

**Inteligência Artificial · Graduação · FACOM/UFMS · Prof. Bruno Nogueira**

Código base do trabalho em grupo. Vocês vão construir um chatbot para um tema do
cotidiano universitário. A cada mensagem, **dois classificadores treinados pelo
grupo** decidem **se** o bot deve responder e **sobre o quê**; depois, o bot
**busca nos documentos oficiais** do tema os trechos mais relevantes e um **LLM**
(acessado pelo OpenRouter, com a chave que o professor envia ao grupo) redige a
resposta a partir desses trechos. Tudo isso é orquestrado por um grafo de estados
([LangGraph](https://langchain-ai.github.io/langgraph/)).

```
                         ┌─ chitchat ─────▶ resposta social (LLM)
mensagem ─▶ [1. escopo] ─┼─ fora_dominio ─▶ recusa educada
                         └─ dominio ─▶ [2. tema] ─┬─ confiança baixa ─▶ "você quis dizer...?"
                                                  └─ intenção X ─▶ busca nos documentos de X
                                                                    ─▶ LLM redige a resposta
```

O **enunciado** (no AVA) tem a explicação completa, passo a passo, com exemplos e
figuras. Este README é o guia rápido do repositório.

---

## 1. Instalação

### No seu computador (recomendado)

Precisa de Python 3.10 ou mais novo e do Git.

```bash
git clone <endereço do repositório do grupo>
cd <pasta do repositório>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                        # abra notebooks/00_exemplo_biblioteca.ipynb
```

### No Google Colab

Abra o notebook pelo GitHub (`Arquivo → Abrir notebook → GitHub`) e edite a
variável `REPO_URL` na primeira célula: ela clona o repositório e instala as
dependências. Os arquivos criados no Colab **somem** quando a sessão termina - faça
commit/push do que importa (dados, documentos, modelos).

## 2. A chave do OpenRouter

O professor envia **uma chave por grupo**. Ela dá acesso a LLMs pagos com os
créditos do grupo, então é **secreta**.

* **No computador:** copie `.env.exemplo` para `.env` e cole a chave:

  ```
  OPENROUTER_API_KEY=sk-or-v1-...a chave do grupo...
  ```

  O `.env` está no `.gitignore`: ele **não** vai para o GitHub. Nunca cole a chave em
  notebooks, no código ou em mensagens públicas. Chave vazada? Avise o professor
  imediatamente.
* **No Colab:** ícone de chave 🔑 na barra lateral (*Secrets*) → *Adicionar novo
  secret* → nome `OPENROUTER_API_KEY`, valor = a chave → ative *Acesso ao notebook*.
* **Teste:** `python scripts/testar_llm.py`.

Sem chave, tudo continua funcionando com as **respostas fixas** do `grupo.yaml`; só
as respostas redigidas pelo LLM ficam desligadas. A **avaliação automática roda
sempre sem LLM**.

## 3. Primeiro contato (15 minutos)

```bash
# converse com o bot de exemplo (tema: biblioteca fictícia)
python scripts/conversar.py --config exemplos/biblioteca/grupo.yaml --depurar

# teste a busca nos documentos do exemplo (não gasta créditos)
python scripts/indexar_documentos.py --config exemplos/biblioteca/grupo.yaml --buscar "posso renovar pelo celular?"

# rode a avaliação padronizada no exemplo
python scripts/avaliar.py --config exemplos/biblioteca/grupo.yaml
```

Depois, leia e execute `notebooks/00_exemplo_biblioteca.ipynb`: ele percorre todas
as etapas do trabalho no tema de exemplo.

> Os modelos de exemplo foram salvos com uma versão específica do scikit-learn. Se
> aparecer erro ou aviso de versão ao carregá-los, execute o notebook 00 - ele
> treina e salva os modelos de novo com a sua versão.

---

## 4. Estrutura do repositório

```
.
├── grupo.yaml                  ← 🔨 configuração: tema, intenções, respostas, LLM, base
├── .env.exemplo                ← modelo do .env (a chave do OpenRouter vai no .env)
├── USO_DE_IA.md                ← 🔨 registro de TODO uso de IA (obrigatório; ver enunciado)
├── dados/
│   ├── treino.csv              ← 🔨 frases de treino (texto,intencao,origem)
│   ├── teste.csv               ← 🔨 frases de teste (escritas por PESSOAS)
│   └── README.md               ← formato e regras dos dados
├── documentos/                 ← 🔨 base de conhecimento (PDF/MD/TXT/HTML) + FONTES.md
├── modelos/                    ← 🔨 escopo.joblib e tema.joblib (gerados pelo notebook)
├── notebooks/
│   ├── 00_exemplo_biblioteca.ipynb   ← exemplo resolvido, já executado
│   └── 01_trabalho_grupo.ipynb       ← 🔨 notebook entregável do grupo
├── chatbot/                    ← código base (pacote Python)
│   ├── rotulos.py              ← rótulos reservados: dominio, chitchat, fora_dominio, esclarecer
│   ├── config.py               ← leitura/validação do grupo.yaml
│   ├── dados.py                ← leitura/validação dos CSVs
│   ├── texto.py                ← normalização e stopwords
│   ├── representacoes.py       ← TF-IDF de palavras/caracteres, embeddings (opcional)
│   ├── modelos.py              ← salvar/carregar classificadores (o "contrato")
│   ├── conhecimento.py         ← documentos → trechos → índice vetorial → busca (RAG)
│   ├── llm.py                  ← cliente do OpenRouter e os prompts do bot
│   ├── geracao.py              ← geração de frases de treino com LLM
│   ├── grafo.py                ← o chatbot em LangGraph
│   └── avaliacao.py            ← métricas padronizadas da correção
├── scripts/
│   ├── testar_llm.py           ← confere a chave do OpenRouter
│   ├── conversar.py            ← chat no terminal (--depurar mostra rotas e trechos)
│   ├── indexar_documentos.py   ← cria o índice e testa buscas nos documentos
│   ├── gerar_dados_llm.py      ← gera frases de treino com o LLM (com registro dos prompts)
│   ├── baixar_massive.py       ← negativos fáceis em português (base MASSIVE)
│   ├── avaliar.py              ← avaliação padronizada (a mesma da correção)
│   └── verificar_entrega.py    ← checklist automático da entrega
├── exemplos/biblioteca/        ← tema de exemplo completo (não pode ser escolhido)
├── relatorio/                  ← 🔨 relatorio.pdf (formato SBC, até 10 páginas)
├── resultados/                 ← saída do scripts/avaliar.py
├── LINKS.txt                   ← 🔨 link do vídeo
├── requirements.txt
└── requirements-extra.txt      ← opcional: sentence-transformers (embeddings)
```

🔨 = o grupo preenche.

---

## 5. Como cada peça é usada

| Quero... | Comando / onde |
|---|---|
| conferir se a chave funciona | `python scripts/testar_llm.py` |
| gerar frases de treino com o LLM | `python scripts/gerar_dados_llm.py` → revisar `dados/gerados_llm.csv` |
| ver se os documentos respondem uma pergunta | `python scripts/indexar_documentos.py --buscar "pergunta"` |
| treinar e comparar classificadores | `notebooks/01_trabalho_grupo.ipynb` |
| conversar com o bot | `python scripts/conversar.py --depurar` (ou `--sem-llm`) |
| medir o desempenho como o professor | `python scripts/avaliar.py` |
| saber se a entrega está pronta | `python scripts/verificar_entrega.py` |

---

## 6. O contrato com a correção automática

O professor corrige todos os grupos com o mesmo script (`scripts/avaliar.py`),
rodando o bot de cada grupo no teste do próprio grupo **e em conjuntos de teste que
não serão divulgados**, **sem LLM** (respostas fixas). Para isso funcionar:

1. **`grupo.yaml`** com a estrutura do modelo (nomes das chaves), de 5 a 10
   intenções com nomes em `minusculas_sem_acento`.
2. **`dados/treino.csv` e `dados/teste.csv`** com as colunas `texto,intencao,origem`.
3. **`modelos/escopo.joblib` e `modelos/tema.joblib`**: objetos com `predict` (e, de
   preferência, `predict_proba`) que recebem **uma lista de textos crus**. O de
   escopo tem exatamente as classes `dominio`, `chitchat` e `fora_dominio`; o de
   tema, exatamente as intenções do `grupo.yaml`. Um `Pipeline` do scikit-learn
   salvo com `chatbot.modelos.salvar_classificadores` cumpre tudo isso.
4. **`chatbot.grafo.construir_grafo(config, clf_escopo, clf_tema)`** devolve o grafo
   compilado, e `grafo.invoke({"mensagem": texto})` termina com `rota` e `resposta`
   no estado. **A rota não pode depender do LLM.**
5. **Não altere** `scripts/avaliar.py`, `scripts/verificar_entrega.py` nem
   `chatbot/avaliacao.py`.
6. Toda biblioteca extra que o grupo usar entra no `requirements.txt`.

Funções e classes próprias usadas dentro de um `Pipeline` devem ficar em um módulo
**dentro de `chatbot/`** - função definida no notebook não pode ser recarregada pelo
`joblib`.

**Antes de entregar:** `python scripts/verificar_entrega.py` deve terminar com
**0 erro(s)**. Faça commit de tudo (dados, documentos, modelos, relatório) e gere o
`.zip` com `git archive -o entrega.zip HEAD` - assim o `.env` com a chave fica de fora.

---

## 7. Uso de IA generativa

É **permitido e incentivado** - inclusive para gerar dados, escrever código e
revisar texto. Mas **todo** uso precisa ser declarado em detalhe numa **seção
própria do artigo** e na **apresentação (vídeo)**. Uso não declarado é tratado
como **plágio: nota zero** no trabalho. Registrem cada uso no `USO_DE_IA.md` desde o
primeiro dia.

---

## 8. Perguntas frequentes

**Posso usar outro algoritmo além dos vistos em aula?** Pode, como comparação
adicional, desde que os três algoritmos obrigatórios de cada estágio sejam da
disciplina.

**Posso usar o LLM para classificar?** Só como **comparação** opcional
(`chatbot.llm.ClassificadorLLM`). O bot entregue usa os classificadores treinados.

**Posso usar embeddings pré-treinados?** Pode, nos classificadores
(`EmbeddingsSentenca`) e/ou na busca (`conhecimento.representacao: embeddings`),
após `pip install -r requirements-extra.txt`. Se usar no modelo final, copie a
dependência para o `requirements.txt`.

**Quanto custa o LLM?** Com os modelos baratos indicados, centenas de mensagens
custam centavos de dólar. A chave tem um limite de créditos definido pelo professor;
os scripts mostram, ao final, quantas chamadas e tokens foram gastos. Evitem laços
que chamem o LLM milhares de vezes (por exemplo, avaliar o `ClassificadorLLM` em
todo o treino).

**Mudei um documento. Preciso fazer algo?** Não: o índice em `.vectordb/` é recriado
automaticamente na próxima conversa.

**O verificador acusa erro no exemplo da biblioteca.** É esperado: o exemplo é
pequeno e todo gerado por LLM, de propósito. Os mínimos valem para o trabalho do
grupo.

---

## Créditos

* Exemplo da biblioteca: frases e documentos escritos para este trabalho
  (universidade fictícia).
* MASSIVE (opcional, via `scripts/baixar_massive.py`): FitzGerald, J. et al.
  *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51
  Typologically-Diverse Languages*. ACL 2023. Licença CC BY 4.0.
