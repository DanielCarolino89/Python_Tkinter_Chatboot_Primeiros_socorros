import spacy
import random
import nltk
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from base import base_conhecimento

# Setup inicial
nlp = spacy.load('pt_core_news_sm')
nltk.download('punkt')

# Frases de saudação
palavras_chave_saudacao = ['oi', 'olá', 'e aí', 'bom dia', 'boa tarde', 'boa noite', 'ajuda', 'socorro', 'primeiros socorros', 'emergência', 'urgência']
respostas_saudacao = [
    'Oi! Como posso te ajudar com primeiros socorros?',
    'Olá! Me pergunte algo sobre primeiros socorros.',
    'Estou aqui para ajudar. O que você quer saber sobre primeiros socorros?'
]

def verificar_saudacao(texto):
    texto = texto.lower()
    for palavra in palavras_chave_saudacao:
        if palavra in texto:
            return random.choice(respostas_saudacao)
    return None

def preprocessar(frase):
    tokens = [token.text for token in nlp(frase.lower()) if not (
        token.is_stop or token.like_num or token.is_punct or token.is_space or len(token) == 1)]
    return ' '.join(tokens)

def gerar_resposta(pergunta, limite=0.2):
    saudacao = verificar_saudacao(pergunta)
    if saudacao:
        return saudacao

    conhecimento_processado = [preprocessar(texto) for texto in base_conhecimento]
    conhecimento_processado.append(preprocessar(pergunta))

    vetorizar = TfidfVectorizer()  #transformar um conjunto de textos em uma matriz
    matriz_tfidf = vetorizar.fit_transform(conhecimento_processado)

    similaridade = cosine_similarity(matriz_tfidf[-1], matriz_tfidf)
    indice_resposta = similaridade.argsort()[0][-2]

    if similaridade[0][indice_resposta] < limite:
        return "Desculpe, não encontrei uma resposta clara sobre isso nos materiais disponíveis."
    else:
        return base_conhecimento[indice_resposta]

# Interface gráfica com Tkinter
def enviar_mensagem():
    pergunta_usuario = entrada_usuario.get()
    if pergunta_usuario.strip() == '':
        return

    janela_conversa.config(state=tk.NORMAL)
    janela_conversa.insert(tk.END, f"\n")
    janela_conversa.insert(tk.END, f"Você: {pergunta_usuario}\n")
    entrada_usuario.delete(0, tk.END)

    if pergunta_usuario.lower() in ['sair', 'exit', 'fim', 'tchau', 'adeus', 'obrigado', 'valeu']:
        resposta = "Até logo! Espero ter ajudado. 🫶"
        janela_conversa.insert(tk.END, f"\n")
        janela_conversa.insert(tk.END, f"DoutorBot: {resposta}\n")
        janela_conversa.config(state=tk.DISABLED)
        raiz.after(2000, raiz.destroy)  # Fecha a janela após 2 segundos
        return

    resposta = gerar_resposta(pergunta_usuario)
    janela_conversa.insert(tk.END, f"\n")
    janela_conversa.insert(tk.END, f"DoutorBot: {resposta}\n")
    janela_conversa.config(state=tk.DISABLED)
    janela_conversa.yview(tk.END)

# Criar janela principal
raiz = tk.Tk()
raiz.title("Guia de Primeiros Socorros")

# Área de conversa
janela_conversa = scrolledtext.ScrolledText(raiz, state=tk.DISABLED, width=60, height=20, wrap=tk.WORD, font=("Arial", 11))
janela_conversa.pack(padx=10, pady=10)

# Campo de entrada
rotulo_pergunta = tk.Label(raiz, text="Digite sua pergunta:", font=("Arial", 12))
rotulo_pergunta.pack(anchor='w', padx=10)
entrada_usuario = tk.Entry(raiz, width=50, font=("Arial", 12))
entrada_usuario.pack(side=tk.LEFT, padx=(10, 0), pady=(0, 10), expand=True, fill=tk.X)

# Botão de envio
botao_enviar = tk.Button(raiz, text="Enviar", command=enviar_mensagem, font=("Arial", 11))
botao_enviar.pack(side=tk.RIGHT, padx=10, pady=(0, 10))

# Mensagens iniciais
janela_conversa.config(state=tk.NORMAL)
janela_conversa.insert(tk.END, "=================================================\n")
janela_conversa.insert(tk.END, "\t🩺 Olá! Bem-vindo ao Guia de Primeiros Socorros.\n")
janela_conversa.insert(tk.END, "\tSou DoutorBot me pergunte algo\n") 
janela_conversa.insert(tk.END, "\tou digite 'sair' para encerrar a conversa.\n")
janela_conversa.insert(tk.END, "=================================================\n")

janela_conversa.config(state=tk.DISABLED)

# Iniciar interface
raiz.mainloop()
