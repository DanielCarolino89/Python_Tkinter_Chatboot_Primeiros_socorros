import speech_recognition as sr

reconhecedor = sr.Recognizer()

with sr.Microphone() as fonte:
    print("Fale algo...")
    audio = reconhecedor.listen(fonte)

try:
    texto = reconhecedor.recognize_google(audio, language="pt-BR")
    print("Você disse:", texto)
except sr.UnknownValueError:
    print("Não entendi o que você disse.")
except sr.RequestError as e:
    print("Erro ao acessar o serviço de reconhecimento:", e)
