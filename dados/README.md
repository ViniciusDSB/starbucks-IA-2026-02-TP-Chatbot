# Dados do grupo

| arquivo      | conteúdo                                                        |
|--------------|-----------------------------------------------------------------|
| `treino.csv` | tudo o que pode ser usado para treinar, validar e ajustar modelos |
| `teste.csv`  | **só** para a avaliação final - não use para escolher modelos   |
| `geral/`     | bases auxiliares (ex.: MASSIVE, via `scripts/baixar_massive.py`) |

## Formato

CSV com cabeçalho, UTF-8, separado por vírgula (textos com vírgula entre aspas):

```
texto,intencao,origem
"oi, tudo bem?",chitchat,humano
como renovo meu livro pelo celular?,renovacao,llm
```

* `intencao`: uma das intenções do `grupo.yaml`, `chitchat` ou `fora_dominio`;
* `origem`: `humano`, `coletado`, `llm`, `aumentado` ou `massive`.

## Regras (ver enunciado)

* **Treino**: no mínimo 50 frases por intenção, 100 de `chitchat` e 150 de
  `fora_dominio`; pelo menos 20% escritas por pessoas (`humano`) ou coletadas
  (`coletado`). Frases geradas por LLM são permitidas (`origem = llm`), desde que
  o modelo e os *prompts* sejam descritos no relatório.
* **Teste**: no mínimo 15 frases por intenção, 30 de `chitchat` e 40 de
  `fora_dominio`, **todas** escritas por pessoas ou coletadas - de preferência por
  quem não é do grupo (colegas, familiares, um formulário online). Nenhuma frase
  do teste pode estar no treino.
* Inclua no `fora_dominio` **negativos difíceis**: perguntas sobre assuntos
  *vizinhos* ao seu tema, que um classificador ingênuo confundiria com o domínio.

`python scripts/verificar_entrega.py` confere tudo isso.
