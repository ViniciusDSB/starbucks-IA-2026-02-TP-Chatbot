"""Rótulos reservados do trabalho.

Estes nomes fazem parte do *contrato* com o script de avaliação: não os altere.

Cada mensagem dos arquivos de dados tem UMA coluna de rótulo, ``intencao``, que
assume um destes valores:

* o nome de uma das intenções do seu domínio (definidas em ``grupo.yaml``);
* ``chitchat``      - conversa social: saudação, agradecimento, despedida, elogio...;
* ``fora_dominio``  - pedidos legítimos, mas que o bot não sabe responder.

O rótulo do **classificador de escopo** (1º estágio) é derivado automaticamente:
qualquer intenção do domínio vira ``dominio``.
"""

DOMINIO = "dominio"
CHITCHAT = "chitchat"
FORA_DOMINIO = "fora_dominio"

#: Rótulo que o grafo usa quando a confiança no tema é baixa demais.
#: Ele nunca aparece nos dados: só pode ser *produzido* pelo chatbot.
ESCLARECER = "esclarecer"

#: Classes do classificador de escopo, nesta ordem (usada nas matrizes de confusão).
CLASSES_ESCOPO = [DOMINIO, CHITCHAT, FORA_DOMINIO]

#: Nomes que não podem ser usados como intenção do domínio.
RESERVADOS = {DOMINIO, CHITCHAT, FORA_DOMINIO, ESCLARECER}

#: Valores aceitos na coluna ``origem`` dos arquivos de dados.
ORIGENS = {
    "humano",     # escrita por uma pessoa (integrante do grupo ou voluntário)
    "coletado",   # retirada de uma fonte real (fórum, FAQ, e-mail anonimizado...)
    "llm",        # gerada por um modelo de linguagem (declare o modelo e o prompt no relatório)
    "aumentado",  # variação automática de outra frase (sinônimos, erros de digitação...)
    "massive",    # base MASSIVE (Amazon), baixada com scripts/baixar_massive.py
}
