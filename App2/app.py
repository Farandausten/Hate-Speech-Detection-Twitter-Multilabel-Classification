import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
app = Flask(__name__)
model = keras.models.load_model('newmodel.h5')
tks = load('newtokens.save')

alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
id_stopword_dict = pd.read_csv('abusive.csv', header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})

factory = StemmerFactory()
stemmer = factory.create_stemmer()
alay_dict_map = dict(zip(alay_dict[0], alay_dict[1]))

def lowercase(text):
    text = text.lower()
    return text

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',str(text)) # Remove every '\n'
    text = re.sub('rt',' ',str(text)) # Remove every retweet symbol
    text = re.sub('user',' ',str(text)) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',str(text)) # Remove every URL
    text = re.sub('  +', ' ', str(text)) # Remove extra spaces
    return text
    
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', str(text)) 
    text = re.sub(r'\d+', ' ', str(text))
    return text

def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])
  
def remove_stopword(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def stemming(text):
    return stemmer.stem(text)

def preprocess(text):
        text = lowercase(text)
        text = remove_nonaplhanumeric(text)
        text = remove_unnecessary_char(text)
        text = normalize_alay(text)
        text = stemming(text) 
        text = remove_stopword(text)
        return text
             

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/index1.html')
def index():
    return render_template('index1.html')

@app.route('/index1.html/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    txt = [x for x in request.form.values()]
    comment = preprocess(txt[0])
    print(txt)
    comment_seq = tks.texts_to_sequences([comment])
    req = pad_sequences(comment_seq, maxlen=142, padding='post', truncating='post')
    p = model.predict(req)
    output=np.round(p)

    category=['Hate Speech','Abusive','Individual','Group','Religion','Race','Physical','Gender','Other','Weak','Moderate','Strong']
    hasil=[]
    for i in range(len(output[0])):
        if output[0][i]==1:
                hasil.append(category[i])

    if len(hasil)==0:
        pred_text='Kalimat bukan berkategori hate speech atau abusive'
    elif category[0] in hasil and category[1] not in hasil:
        pred_text='Kalimat berkategori Hate Speech tipe '+', '.join(hasil[1:])
    elif category[0] not in hasil and category[1] in hasil:
        pred_text="Kalimat berkategori Abusive"
    elif category[0] in hasil and category[1] in hasil:
        pred_text='Kalimat berkategori Abusive dan Hate Speech tipe '+', '.join(hasil[2:])
        
    return render_template('index1.html', TXT=txt[0], predict_text=pred_text)    



if __name__ == "__main__":
    app.run(debug=True)