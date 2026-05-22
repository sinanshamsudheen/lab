#------------OHE--------------
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

sentence = input("Enter sentence: ")
word_token = word_tokenize(sentence)

filter = []
for word in word_token:
  if word.isalpha() and word not in stop_words:
    filter.append(word.lower())

vocab = sorted(set(filter))

ohe_vector = {}
for word in vocab:
  vector = [1 if word == token else 0 for token in vocab]
  ohe_vector[word] = vector

print("\nOne Hot Encoding:")
for word,vector in ohe_vector.items():
  print(f"{word} : {vector}")

#-------------BOW------------
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
stopwords = set(stopwords.words('english'))

document = []
for i in range(3):
  text = input(f"Enter sentence {i}: ")
  document.append(text)

pro_doc = []
for doc in document:
  doc = doc.lower()
  tokens = word_tokenize(doc)
  filter = []
  for word in tokens:
    if word.isalpha() and word not in stopwords:
      filter.append(word)
  pro_doc.append(filter)

vocab = []
for doc in pro_doc:
  for word in doc:
    if word not in vocab:
      vocab.append(word)

bow = []
for doc in pro_doc:
  count = Counter(doc)
  vector = [count[word] for word in vocab]
  bow.append(vector)

print(f"Vocabulary is: {vocab}")
print("Bag of words is:")
for vec in bow:
  print(vec)

#------------TFIDF--------------------
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords = set(stopwords.words('english'))

document = []
for i in range(3):
  text = input(f"Enter sentence {i}: ")
  document.append(text)

clean_doc = []
for doc in document:
    text = []
    doc = doc.lower()
    token = word_tokenize(doc)
    for word in token:
      if word.isalpha() and word not in stopwords:
        text.append(word)
    clean_doc.append(" ".join(text))

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(clean_doc)

print("IDF Values:")
idf = vectorizer.idf_
words = vectorizer.get_feature_names_out()
for i in range(len(words)):
  print(f"{words[i]} : {idf[i]}")

print("\nTF Values:")
tf_mat = matrix.toarray()
for i in range(len(clean_doc)):
  for j in range(len(words)):
    tfidf = tf_mat[i][j]
    if tfidf > 0:
      tf = tf_mat[i][j] / idf[j]
      print(f"{words[j]} : {tf}")

print(f"\nTF-IDF Matrix:\n {matrix.toarray()}")

#-----------------COSINE-----------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = "/content/text.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    sentences = file.readlines()

query = input("Enter your query: ")

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([query]+sentences)

similarity = cosine_similarity(tfidf[0],tfidf[1:])[0]

best_index = similarity.argmax()

print(f"Most similar sentence: {sentences[best_index]}")
print(f"Similarity: {similarity[best_index]*100}")

#---------------SENTIMENT--------------
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

df = pd.read_csv("/content/IMDB-Dataset.csv")
df['sentiment'] = df['sentiment'].str.strip().str.lower()
label_map = {"negative":0,"positive":1}
lemma = WordNetLemmatizer()
stemm = PorterStemmer()

pro_doc = []
for text in df['review']:
  tokens = word_tokenize(text.lower())
  filter = []
  for word in tokens:
    if word.isalpha() and word not in stopwords:
      lemmatized = lemma.lemmatize(word)
      stemmed = stemm.stem(lemmatized)

      filter.append(stemmed)
  pro_doc.append(" ".join(filter))
df['pro_text'] = pro_doc

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['pro_text'])
y = df['sentiment'].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = MultinomialNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy: {accuracy}")

user = input("Enter text: ")
user_token = word_tokenize(user)
filtered = []
for word in user_token:
  if word.isalnum() and word not in stopwords:
    filtered.append(word)

final = " ".join(filtered)

vector = vectorizer.transform([final])
prediction = model.predict(vector)[0]

reverse_map = {0:"negative",1:"positive"}
print(f"Prediction: {reverse_map[prediction]}")

#------------NER-------------------
import nltk
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag

user_input = input("Enter a sentence: ")

tokens = word_tokenize(user_input)
tags = pos_tag(tokens)
ner_tree = ne_chunk(tags)

for subtree in ner_tree:
  if hasattr(subtree, 'label'):
    entity_name = " ".join([token for token,pos in subtree.leaves()])
    entity_type = subtree.label()
    print(f"{entity_name} : {entity_type}")

#--------------Chatbot-------------
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read dataset
with open("/content/text.txt", "r") as file:
    corpus = file.readlines()

print("Chatbot Ready! Type 'exit' to quit.\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("Bot: Bye!")
        break

    # Add user query temporarily
    data = corpus + [query]

    # Convert text to vectors
    tfidf = TfidfVectorizer().fit_transform(data)

    # Compare query with dataset
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])

    # Get best matching sentence
    index = similarity.argmax()

    print("Bot:", corpus[index])


#----------------Antonym and Synonym---------------
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

word = input("Enter a word: ").lower()

synonyms = set()
antonyms = set()

for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
        # Synonyms
        synonyms.add(lemma.name())
        # Antonyms
        if lemma.antonyms():
            antonyms.add(lemma.antonyms()[0].name())

print("\nSynonyms:")
for s in synonyms:
    print(s)

print("\nAntonyms:")
for a in antonyms:
    print(a)

#-----------Replace with antonym/synonym-----------
# Positive words -> synonym replacement
synonyms = {
    "good": "excellent",
    "happy": "joyful",
    "fast": "quick",
    "beautiful": "pretty",
    "smart": "intelligent"
}
# Negative words -> antonym replacement
antonyms = {
    "bad": "good",
    "sad": "happy",
    "slow": "fast",
    "ugly": "beautiful",
    "hate": "love"
}

# Negation words
negations = ["not", "no", "never"]
sentence = input("Enter sentence: ").lower()
words = sentence.split()
result = []
i = 0

# Sliding window traversal
while i < len(words):
    word = words[i]
    # Check negation + next word
    if word in negations and i + 1 < len(words):
        next_word = words[i + 1]
        # not good -> bad
        if next_word in synonyms:
            for key, value in antonyms.items():
                if value == next_word:
                    result.append(key)
                    break
            i += 2
            continue
        # not bad -> excellent/good
        elif next_word in antonyms:
            result.append(antonyms[next_word])
            i += 2
            continue
    # Normal positive replacement
    if word in synonyms:
        result.append(synonyms[word])
    # Normal negative replacement
    elif word in antonyms:
        result.append(antonyms[word])
    else:
        result.append(word)
    i += 1

print("\nModified sentence:")
print(" ".join(result))

#---------------HMM---------------
import nltk
from nltk.tag import hmm
from nltk.corpus import treebank

train_data = treebank.tagged_sents()[:3000]
trainer = hmm.HiddenMarkovModelTrainer()
model = trainer.train_supervised(train_data)

sentence = input("Enter a sentence: ")
word = sentence.split()

tagged = model.tag(word)
print("Pos Tags: ",tagged)

#-----------------MT-----------------------
import nltk
from nltk.translate import IBMModel1
from nltk.translate import AlignedSent
from nltk.tokenize import word_tokenize
bitext = [
  AlignedSent(["i", "love", "python"], ["je", "aime", "python"]),
  AlignedSent(["i", "love", "books"], ["je", "aime", "livres"]),
  AlignedSent(["this", "is", "a", "book"], ["ceci", "est", "un", "livre"]),
  AlignedSent(["i", "am", "a", "student"], ["je", "suis", "un", "etudiant"]),
  AlignedSent(["good", "morning"], ["bonjour"]),
  AlignedSent(["how", "are", "you"], ["comment", "allez", "vous"]),
  AlignedSent(["i", "eat", "apple"], ["je", "mange", "pomme"]),
  AlignedSent(["she", "reads", "a", "book"], ["elle", "lit", "un", "livre"]),
  AlignedSent(["he", "likes", "music"], ["il", "aime", "musique"]),
]

ibm1 = IBMModel1(bitext, 50)
sentence = input("Enter English sentence: ")
words = word_tokenize(sentence.lower())

translated = []
for word in words:
  translations = ibm1.translation_table.get(word, {})
  best_word = max(translations, key=translations.get, default=word)
  translated.append(best_word)

print("\nTranslated sentence:")
print(" ".join(translated))
