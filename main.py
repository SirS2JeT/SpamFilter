import os
import email
import random
import email.policy
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt

base_directory = "Data/"
spam_email_names = os.listdir(base_directory + "spam")
normal_email_names = os.listdir(base_directory + "ham")

def load_email(is_spam, filename):
    directory = base_directory + ("spam" if is_spam else "ham")
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

spam_emails = [load_email(True, filename) for filename in spam_email_names]
normal_emails = [load_email(False, filename) for filename in normal_email_names]
random.shuffle(spam_emails)
random.shuffle(normal_emails)

def process_email(emails, label, data_dictionary, default_topic=None):
    for mail in emails:
        payload = mail.get_payload()
        if isinstance(payload, list):
            process_email(payload, label, data_dictionary, default_topic=mail["Subject"])
        else:
            if "Content-Type" in mail.keys():
                if "html" in mail["Content-Type"].lower():
                    try:
                        soup = BeautifulSoup(mail.get_content())
                        topic = mail["Subject"]
                        if topic == None:
                            topic = default_topic
                        content = soup.body.text
                        data_dictionary["topic"].append(topic)
                        data_dictionary["content"].append(content)
                        data_dictionary["label"].append(label)
                    except:
                        pass
                elif "plain" in mail["Content-Type"].lower():
                    try:
                        topic = mail["Subject"]
                        if topic == None:
                            topic = default_topic
                        content = mail.get_content()
                        data_dictionary["topic"].append(topic)
                        data_dictionary["content"].append(content)
                        data_dictionary["label"].append(label)
                    except:
                        pass
                else:
                    pass

data_dictionary = {"topic": [], "content": [], "label": []}
process_email(spam_emails, 1, data_dictionary)
process_email(normal_emails, 0, data_dictionary)
df = pd.DataFrame(data_dictionary)
df.dropna(inplace=True)
df = df.sample(frac=1)

class_weight = 1 / df["label"].value_counts()
class_weight = dict(class_weight / class_weight.sum())

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
def preprocess_text(content):
    # Change text to lowercased
    content = content.lower()
    # Remove stop words
    for stopword in stopwords:
        content = content.replace(stopword + " ", "")
        content = content.replace(" " + stopword, "")
    return content

num_words = 8196
max_content_length = 512
topic_and_contents = []
for (topic, content) in zip(df["topic"], df["content"]):
    topic_and_contents.append(preprocess_text(topic + " " + content))
df["topic_content"] = topic_and_contents
tokenizer = Tokenizer(num_words=num_words - 1, oov_token="<OOV>")
tokenizer.fit_on_texts(topic_and_contents)

tokenized_topic_and_contents = tokenizer.texts_to_sequences(topic_and_contents)

tokenized_topic_and_content_lengths = pd.DataFrame([len(item) for item in tokenized_topic_and_contents])

padding_tokenized_topic_and_contents = pad_sequences(
    tokenized_topic_and_contents,
    padding='post',
    truncating='post',
    maxlen=max_content_length
)

X = np.array(padding_tokenized_topic_and_contents)
y = np.array(df["label"])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.15)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, 256, input_length=max_content_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    class_weight=class_weight,
    epochs=10
)

pd.DataFrame(history.history).plot(xlabel="Epoch")
plt.title("Loss & Accuracy over time")

plt.show()
