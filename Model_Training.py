import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
df = pd.read_csv("fake_job_postings.csv")
df.head()
df.shape
df.info()
df.nunique()
df['fraudulent'].value_counts()
df.isnull().sum()
df['location'].fillna('Unknown', inplace=True)
df['department'].fillna('Unknown', inplace=True)
df['salary_range'].fillna('Not Specified', inplace=True)
df['employment_type'].fillna('Not Specified', inplace=True)
df['required_experience'].fillna('Not Specified', inplace=True)
df['required_education'].fillna('Not Specified', inplace=True)
df['industry'].fillna('Not Specified', inplace=True)
df['function'].fillna('Not Specified', inplace=True)
df.head()
df['combined_text'] = df[['title', 'location', 'salary_range', 'company_profile', 'description', 
                          'requirements', 'benefits', 'employment_type', 
                          'required_experience', 'required_education', 'industry', 
                          'function', 'department']].apply(
    lambda x: ' '.join(x.fillna('').astype(str)), axis=1
)
df.isnull().sum()
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
df[text_columns] = df[text_columns].fillna('Missing')
df.isna().sum()
df.drop(columns=['title',
                 'location',
                 'salary_range',
                 'company_profile',
                 'description',
                 'requirements',
                 'benefits',
                 'employment_type',
                 'required_experience',
                 'required_education',
                 'industry',
                 'function',
                 'department',
                 'job_id',
                 'has_questions',
                 'telecommuting'], inplace=True)

df.head()
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower().strip()  
    return text

# Apply text cleaning
df['combined_text'] = df['combined_text'].apply(clean_text)

le = LabelEncoder()
df['fraudulent'] = le.fit_transform(df['fraudulent'])

df.head()
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df['combined_text'])
sequences = tokenizer.texts_to_sequences(df['combined_text'])

X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

y = df['fraudulent']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Original Dataset Shape:", X.shape, y.shape)
print("After SMOTE Resampling Shape:", X_resampled.shape, y_resampled.shape)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Training LSTM Model...")
lstm_model = Sequential([
    Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_history = lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

plt.plot(lstm_history.history['accuracy'], label='Train Accuracy')
plt.plot(lstm_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Accuracy')
plt.legend()
plt.show()


print("LSTM Model Evaluation")
lstm_predictions = (lstm_model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, lstm_predictions))
print("LSTM F1 Score:", f1_score(y_test, lstm_predictions))

lstm_model.save("fake_job_lstm_model.h5")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)