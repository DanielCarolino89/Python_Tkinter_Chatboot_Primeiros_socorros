# 🩺 DoutorBot - Chatbot de Primeiros Socorros com Voz e Análise de Sentimento.

**DoutorBot** é um assistente virtual interativo de **primeiros socorros** com interface gráfica em Python (Tkinter). Ele responde perguntas via texto ou voz, interpreta o **sentimento do usuário** com IA, e oferece orientações básicas baseadas em uma base de conhecimento personalizada.

---

## 🚀 Funcionalidades

- 🎤 Reconhecimento de voz em tempo real
- 💬 Chat via interface gráfica com Tkinter
- 🧠 Análise de sentimento usando modelo BERT multilingue (Hugging Face)
- 🧾 Respostas automáticas com base em similaridade textual (TF-IDF + NLP)
- 📚 Saudações e mensagens inteligentes baseadas no horário
- 🔄 Processamento em tempo real com threading para escuta contínua

---

## 🖼️ Interface

A interface consiste em:
- Um histórico de conversa
- Entrada de texto para perguntas
- Botões para enviar, ativar escuta por voz e analisar sentimento
- Rodapé com créditos

---

## 🛠️ Tecnologias Utilizadas

- `Python 3.x`
- `Tkinter` — interface gráfica
- `SpeechRecognition` e `PyAudio` — captura e transcrição de áudio
- `transformers` — modelo de sentimento pré-treinado (`nlptown/bert-base-multilingual-uncased-sentiment`)
- `spaCy` — processamento de linguagem natural (`pt_core_news_sm`)

  ---

  ## 📖 Exemplos de Uso

**Perguntas que você pode fazer:**

- “O que faço em caso de queimadura?”
- “Como socorrer alguém desmaiado?”
- “Minha avó caiu, o que devo fazer?”

**Comando de voz:**

Clique no botão **🎤 Iniciar Escuta** e fale sua pergunta diretamente ao DoutorBot.

**Análise de sentimento:**

Após enviar uma pergunta, clique no botão **😊 Sentimento** para que o DoutorBot analise o tom emocional da sua mensagem.

---

## 📌 Observações

- A base de conhecimento está localizada no arquivo `base.py` (importado em `main2.py`).
- Este projeto **não substitui profissionais de saúde** ou serviços de emergência.

---

## ⚠️ Aviso Legal

> ❗ **Este projeto é apenas educacional.**  
> **Não utilize em situações reais de risco.**  
> Em emergências reais, procure ajuda médica ou ligue para os serviços oficiais (como o SAMU - 192 no Brasil).

---

## 👤 Autor

**Daniel Carolino**  
Desenvolvedor Web | Designer Gráfico  

🔗 [LinkedIn](https://linkedin.com/in/danielcarolino)  
📧 [Email](mailto:daniel.carolino@gmail.com)

- `nltk` — tokenização
- `scikit-learn` — TF-IDF e similaridade textual

---


