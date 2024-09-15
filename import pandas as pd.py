import pandas as pd 

# Read the Excel file (adjust the file path)
df = pd.read_excel(r'File destination')

# Assuming your data is in the first sheet (you can specify the sheet name if needed)
for index, row in df.iterrows():
    text_to_speak = row['A']  # Replace 'column_name' with the actual column name

    # Create a gTTS object
    tts = gTTS(text_to_speak)

    # Save the audio file (adjust the path as needed)
    audio_file_path = fr'File destination'
    tts.save(audio_file_path)
