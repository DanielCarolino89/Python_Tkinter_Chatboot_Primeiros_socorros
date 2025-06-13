# Importações de bibliotecas necessárias
import spacy
import random
import nltk
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from base import base_conhecimento

# Hugging Face e reconhecimento de voz
from transformers import pipeline
import speech_recognition as sr
import threading

# Carregamento do modelo de linguagem SpaCy e download de tokenizer do NLTK
nlp = spacy.load('pt_core_news_sm')
nltk.download('punkt')

# Carregamento do pipeline de análise de sentimentos da Hugging Face
sentimento_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Variáveis globais para controlar gravação de áudio e última pergunta do usuário
gravando_audio = False
modo_continuo = False
ultima_pergunta_usuario = ""

# Palavras e frases para detectar saudações
palavras_chave_saudacao = ['oi','ola', 'olá','e ai','e aí','dia','tarde','noite','bom', 'bom dia', 'boa tarde', 'boa noite', 'ajuda', 'socorro', 'primeiros socorros', 'emergência', 'urgência']
respostas_saudacao = [
    'Oi! Como posso te ajudar com primeiros socorros?',
    'Olá! Me pergunte algo sobre primeiros socorros.',
    'Estou aqui para ajudar. O que você quer saber sobre primeiros socorros?'
]

# Verifica se a entrada do usuário contém uma saudação
def verificar_saudacao(texto):
    texto = texto.lower()
    for palavra in palavras_chave_saudacao:
        if palavra in texto:
            return random.choice(respostas_saudacao)
    return None

# Remove stopwords, pontuações e números da frase para processamento
def preprocessar(frase):
    tokens = [token.text for token in nlp(frase.lower()) if not (
        token.is_stop or token.like_num or token.is_punct or token.is_space or len(token) == 1)]
    return ' '.join(tokens)

# Gera resposta baseada na similaridade da pergunta com a base de conhecimento
def gerar_resposta(pergunta, limite=0.2):
    saudacao = verificar_saudacao(pergunta)
    if saudacao:
        return saudacao

    conhecimento_processado = [preprocessar(texto) for texto in base_conhecimento]
    conhecimento_processado.append(preprocessar(pergunta))

    vetorizar = TfidfVectorizer()
    matriz_tfidf = vetorizar.fit_transform(conhecimento_processado)

    similaridade = cosine_similarity(matriz_tfidf[-1], matriz_tfidf)
    indice_resposta = similaridade.argsort()[0][-2]

    if similaridade[0][indice_resposta] < limite:
        return "Desculpe, não encontrei uma resposta clara sobre isso nos materiais disponíveis."
    else:
        return base_conhecimento[indice_resposta]

# Usa modelo Hugging Face para analisar o sentimento de uma frase
def analisar_sentimento(texto):
    resultado = sentimento_pipeline(texto)[0]
    return f"Sentimento detectado: {resultado['label']} (confiança: {resultado['score']:.2f})"

# Inicia ou para a escuta de áudio
def iniciar_escuta():
    global gravando_audio, modo_continuo
    if gravando_audio:
        gravando_audio = False
        botao_voz.config(text="🎤 Iniciar Escuta", bg="SystemButtonFace", fg="black")
        return

    gravando_audio = True
    modo_continuo = True
    botao_voz.config(text="🛑 Parar Escuta", bg="red", fg="white")
    janela_conversa.config(state=tk.NORMAL)
    janela_conversa.tag_config("negrito", font=("Arial", 12, "bold"))
    janela_conversa.insert(tk.END, "\nDoutorBot: \n","negrito")
    janela_conversa.insert(tk.END, "Modo escuta iniciado...\n")
    janela_conversa.config(state=tk.DISABLED)
    janela_conversa.yview(tk.END)

    threading.Thread(target=executar_escuta_continua).start()

# Escuta contínua por áudio e transcreve com Google Speech Recognition
def executar_escuta_continua():
    global gravando_audio
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        while gravando_audio:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
                texto = recognizer.recognize_google(audio, language="pt-BR")
                entrada_usuario.delete(0, tk.END)
                entrada_usuario.insert(0, texto)
                janela_conversa.config(state=tk.NORMAL)
                janela_conversa.tag_config("negrito", font=("Arial", 12, "bold"))
                janela_conversa.insert(tk.END, f"\nDoutorBot: \n", "negrito")
                janela_conversa.insert(tk.END, f"verificando...\n")
                janela_conversa.config(state=tk.DISABLED)
                janela_conversa.yview(tk.END)
                enviar_mensagem() # Envia a mensagem automaticamente
                break
            except sr.WaitTimeoutError:
                atualizar_indicador("\n⌛ Tempo de escuta expirado... aguardando novamente.\n")
                continue
            except sr.UnknownValueError:
                atualizar_indicador("\n⚠️ Não entendi o que você disse.\n") 
                continue

    atualizar_indicador("🔇")
    botao_voz.config(text="🎤 Iniciar Escuta", bg="SystemButtonFace", fg="black")

# Atualiza a interface com mensagens informativas
def atualizar_indicador(texto):
    janela_conversa.config(state=tk.NORMAL)
    janela_conversa.insert(tk.END, texto + "\n")
    janela_conversa.config(state=tk.DISABLED)
    janela_conversa.yview(tk.END)

# Exibe resultado de análise de sentimento da última pergunta
def usar_analise_sentimento():
    if ultima_pergunta_usuario.strip():
        resultado = analisar_sentimento(ultima_pergunta_usuario)
        janela_conversa.config(state=tk.NORMAL)
        janela_conversa.tag_config("verde", font=("Arial", 12, "bold"), foreground="green")
        janela_conversa.insert(tk.END, "\nDoutorBot: \n", "negrito")
        janela_conversa.insert(tk.END, f"\nAnálise de Sentimento da Última Pergunta:\n➡ \"{ultima_pergunta_usuario}\"\n🧠 {resultado}\n", "verde")
        janela_conversa.config(state=tk.DISABLED)
        janela_conversa.yview(tk.END)
    else:
        janela_conversa.config(state=tk.NORMAL)
        janela_conversa.insert(tk.END, "\n⚠️ Nenhuma pergunta encontrada para analisar sentimento.\n", "verde")
        janela_conversa.config(state=tk.DISABLED)
        janela_conversa.yview(tk.END)

# Lida com o envio da pergunta pelo usuário e gera a resposta
def enviar_mensagem():
    global ultima_pergunta_usuario
    pergunta_usuario = entrada_usuario.get()
    if pergunta_usuario.strip() == '':
        return

    ultima_pergunta_usuario = pergunta_usuario  

    janela_conversa.config(state=tk.NORMAL,)
    janela_conversa.tag_config("negrito", font=("Arial", 12, "bold"))
    janela_conversa.insert(tk.END, f"\nVocê: ", "negrito")
    janela_conversa.insert(tk.END, f"{pergunta_usuario}\n")
    entrada_usuario.delete(0, tk.END)

    if pergunta_usuario.lower() in ['sair', 'exit', 'fim', 'tchau', 'adeus', 'obrigado', 'valeu']:
        resposta = "Até logo! Espero ter ajudado."
        janela_conversa.insert(tk.END, f"\nDoutorBot: ", "negrito")
        janela_conversa.insert(tk.END, f"{resposta}\n")
        janela_conversa.config(state=tk.DISABLED)
        raiz.after(1200, raiz.destroy)
        return

    resposta = gerar_resposta(pergunta_usuario)
    janela_conversa.insert(tk.END, f"\nDoutorBot: ", "negrito")
    janela_conversa.insert(tk.END, f"{resposta}\n")
    janela_conversa.config(state=tk.DISABLED)
    janela_conversa.yview(tk.END)

import tkinter as tk
from tkinter import scrolledtext

# Janela principal
raiz = tk.Tk()
raiz.title("Guia de Primeiros Socorros")

# === FRAME PRINCIPAL PARA CONTEÚDO ===
frame_conteudo = tk.Frame(raiz)
frame_conteudo.pack(fill=tk.BOTH, expand=True)

# Caixa de conversa com rolagem
janela_conversa = scrolledtext.ScrolledText(frame_conteudo, state=tk.DISABLED, width=60, height=25, wrap=tk.WORD, font=("Arial", 12))
janela_conversa.pack(padx=30, pady=(20, 10), fill=tk.BOTH, expand=True)

# Entrada de texto e rótulo
frame_entrada = tk.Frame(frame_conteudo)
frame_entrada.pack(fill=tk.X, padx=10)

rotulo_pergunta = tk.Label(frame_entrada, text="Digite sua pergunta:", font=("Arial", 12))
rotulo_pergunta.pack(side=tk.LEFT)

entrada_usuario = tk.Entry(frame_entrada, width=50, font=("Arial", 12))
entrada_usuario.pack(side=tk.LEFT, padx=(10, 0), pady=(0, 10), expand=True, fill=tk.X)
entrada_usuario.bind("<Return>", lambda event: enviar_mensagem())

# Botões de ações (enviar, voz, sentimento)
frame_botoes = tk.Frame(frame_conteudo)
frame_botoes.pack(pady=(0, 10))

botao_enviar = tk.Button(frame_botoes, text="Enviar", command=enviar_mensagem, font=("Arial", 12))
botao_enviar.pack(side=tk.LEFT, padx=5)

botao_voz = tk.Button(frame_botoes, text="🎤 Iniciar Escuta", font=("Arial", 12), command=iniciar_escuta)
botao_voz.pack(side=tk.LEFT, padx=5)

botao_sentimento = tk.Button(frame_botoes, text="😊 Sentimento", command=usar_analise_sentimento, font=("Arial", 12))
botao_sentimento.pack(side=tk.LEFT, padx=5)

# === RODAPÉ ===
frame_rodape = tk.Frame(raiz)
frame_rodape.pack(side=tk.BOTTOM, fill=tk.X)

texto_copy = tk.Label(frame_rodape, text="© 2025 - Desenvolvido por Daniel Carolino", font=("Arial", 8))
texto_copy.pack(pady=4)


# Mensagem de boas-vindas com pausas entre as falas

janela_conversa.config(state=tk.NORMAL)
janela_conversa.tag_config("negrito", font=("Arial", 12, "bold"))
janela_conversa.tag_config("italic", font=("Arial", 12, "italic"))

def mensagem1():
    janela_conversa.config(state=tk.NORMAL)
    janela_conversa.insert(tk.END, "DoutorBot: ", "negrito")
    janela_conversa.insert(tk.END, "Olá! Sou ")
    janela_conversa.insert(tk.END, "DoutorBot ", "italic")
    janela_conversa.insert(tk.END, "seu assistente de Primeiros Socorros.\n")
    janela_conversa.config(state=tk.DISABLED)
    janela_conversa.after(1200, mensagem2)

def mensagem2():
    janela_conversa.config(state=tk.NORMAL)
    janela_conversa.insert(tk.END, "\nDoutorBot: ", "negrito")
    janela_conversa.insert(tk.END, "farei o meu melhor para ajudar com orientações seguras\n")
    janela_conversa.config(state=tk.DISABLED)
    janela_conversa.after(1200, mensagem3)

def mensagem3():
    janela_conversa.config(state=tk.NORMAL)
    janela_conversa.insert(tk.END, "\nDoutorBot: ", "negrito")
    janela_conversa.insert(tk.END, "descreva o que posso te ajudar.\n")
    janela_conversa.config(state=tk.DISABLED)
    janela_conversa.after(1200, mensagem4)

def mensagem4():
    janela_conversa.config(state=tk.NORMAL)
    janela_conversa.insert(tk.END, "\nDoutorBot: ", "negrito")
    janela_conversa.insert(tk.END, "ou digite ")
    janela_conversa.insert(tk.END, "'sair' ", "negrito")
    janela_conversa.insert(tk.END, "para encerrar a conversa.\n")
    janela_conversa.config(state=tk.DISABLED)

# Inicia a sequência de mensagens
mensagem1()

# Inicia a interface
raiz.mainloop()
