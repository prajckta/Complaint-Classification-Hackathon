# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:35:54 2024

@author: Shreeya
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Process each chunk
for i in range(1, 18):
    # Load the CSV chunk
    df_chunk = pd.read_csv(f'D:/datathon/preprocessed_complaints_chunk{i}.csv')
    
    # Tokenize the text
    df_chunk['tokens'] = df_chunk['narrative'].apply(lambda x: word_tokenize(x))
    
    # Save the tokenized data to a new CSV file
    df_chunk.to_csv(f'D:/datathon/tokenized_complaints_chunk{i}.csv', index=False)

print("Tokenization complete")
