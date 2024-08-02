import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')

# Load the CSV file in chunks
chunk_size = 10000
df_chunks = pd.read_csv('D:/datathon/complaints.csv', chunksize=chunk_size)

# Initialize stemmer
stemmer = PorterStemmer()

# Process each chunk
for i, df_chunk in enumerate(df_chunks):
    print(f"Processing chunk {i+1}")
    
    # Remove duplicates and missing values
    df_chunk = df_chunk.drop_duplicates().dropna()
    
    # Convert text to lowercase and perform stemming
    df_chunk['narrative'] = df_chunk['narrative'].str.lower().apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))
    
    # Save the preprocessed chunk to a new CSV file
    df_chunk.to_csv(f'D:/datathon/preprocessed_complaints_chunk{i+1}.csv', index=False)

print("Processing complete")
