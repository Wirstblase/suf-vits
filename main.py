import requests

def get_speaker_embedding(audio_file_path):
    url = 'http://localhost:5222/get_embedding'
    files = {'file': open(audio_file_path, 'rb')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        embedding = response.json()['embedding']
        return embedding
    else:
        print(f"Error: {response.json()['error']}")
        return None

# Example usage
embedding = get_speaker_embedding('samples/suflea_sample.wav')
print(embedding)
