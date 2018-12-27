from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os
import pyttsx3
import speech_recognition as sr
r=sr.Recognizer()
engine= pyttsx3.init()
question=['is','am','are','has','have','will','shall','has','have','had','do','does','did','where','why','when','who','how','what','which']

bot = ChatBot('Bot', storage_adapter='chatterbot.storage.SQLStorageAdapter',
              logic_adapters=[
                      {
                              'import_path':'chatterbot.logic.BestMatch'
                        },
                      {
                              'import_path':'chatterbot.logic.LowConfidenceAdapter',
                              'threshold':0.70,
                              'default_response':'I am sorry, but I do not understand.'
                        },
                      ],
              trainer='chatterbot.trainers.ListTrainer')

while True:
        message='hi'
        n=input("wana speak: (y/n)")
        if n=='y':
                say='sorry could not recognize your voice'
                with sr.Microphone() as source:
                        print('Say')
                        audio=r.listen(source,phrase_time_limit=2)
                        try:
                                message=r.recognize_google(audio)
                                if message.split(" ")[0] in question:message=message+"?"
                                print('You said: '+message)
                        except:pass
        else:
                message=input('user: ')

        if message.strip() != 'bye':
                reply = bot.get_response(message)
                say=str(reply).strip()
                print('Zetron :', str(reply).strip("-"))
        else:
                print("Zetron: Good Bye")
                break
        engine.say(say)
        engine.runAndWait()
	
'''
bot=ChatBot('Bot',
            storatge_adapter='chatterbot.storage.SQLStorageAdapter',
            trainer='chatterbot.trainers.ListTrainer')
for file in os.listdir('data/'):
        data=open('data/'+file,'r').readlines()
        bot.set_trainer(ListTrainer)
        bot.train(data)

'''


