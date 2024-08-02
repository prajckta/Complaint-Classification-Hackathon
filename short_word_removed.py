import pandas as pd

# Process each tokenized chunk
for i in range(1, 18):
    # Load the tokenized CSV file
    df_chunk = pd.read_csv(f'D:/datathon/tokenized_complaints_chunk{i}.csv')
    
    # Filter out short words (length < 3)
    df_chunk['tokens'] = df_chunk['tokens'].apply(lambda x: [word for word in eval(x) if len(word) >= 3])
    
    # Save the updated tokenized data to a new CSV file
    df_chunk.to_csv(f'D:/datathon/filtered_tokens_complaints_chunk{i}.csv', index=False)

print("Short word removal complete")
