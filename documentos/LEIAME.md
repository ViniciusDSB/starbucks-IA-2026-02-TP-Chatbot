# Base de conhecimento do grupo

Coloquem aqui os **documentos oficiais** do tema - é deles que o bot tira as
respostas quando o LLM está ligado (RAG). Formatos aceitos: `.pdf`, `.md`, `.txt`,
`.html`. Este arquivo (`LEIAME.md`) e o `FONTES.md` não entram na base.

```
documentos/
├── FONTES.md            ← obrigatório: de onde veio cada documento
├── geral/               ← vale para qualquer intenção (regimento, FAQ geral, contatos)
├── <intencao_1>/        ← uma pasta com o NOME EXATO de cada intenção do grupo.yaml
├── <intencao_2>/
└── ...
```

* Mínimo de **5 documentos** no total; o ideal é cada intenção ter o seu.
* Bons documentos: regulamentos, resoluções, editais, páginas de perguntas
  frequentes, manuais (baixe as páginas como PDF ou HTML, ou copie o texto para
  `.md`). Podem ser documentos reais da UFMS (citem a URL) ou textos escritos pelo
  grupo, desde que plausíveis e coerentes com as respostas do `grupo.yaml`.
* PDF escaneado (imagem) não tem texto extraível: prefira a versão digital.
* Depois de mudar os documentos, nada a fazer: o índice em `.vectordb/` é recriado
  sozinho. Para testar a busca sem gastar créditos:
  `python scripts/indexar_documentos.py --buscar "sua pergunta"`.
* Veja o exemplo completo em `exemplos/biblioteca/documentos/`.
