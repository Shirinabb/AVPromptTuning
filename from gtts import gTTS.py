from gtts import gTTS 

# Text that you want to convert to speech
text_to_speak = 'hello'

# Create a gTTS object
tts = gTTS(text_to_speak)

# Save the audio file, make sure to escape backslashes or use raw string notation for the file path
audio_file_path = r'file destination'

# Save the audio file
tts.save(audio_file_path)
