import json
import time
import cv2
import numpy as np
import spotipy
from keras.models import model_from_json
import openai
from spotipy.oauth2 import SpotifyOAuth


openai.api_key = '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'

CLIENT_ID = 'af87c994777b47bb92ab0a1622b7478a'
CLIENT_SECRET = 'aac47d8940474a1a9721493efdece40c'
REDIRECT_URI = 'http://localhost:8080'


def music_recommendation(emotion_output):
    emotion_str = ', '.join([f"{key}: {value}" for key, value in emotion_output.items()])

    prompt = f'I am feeling {emotion_str}. Suggest me 10 songs. The answer you give should be in JSON format with title and artist in double quotes only. Do not write anything else in the answer other than the music list. Example:"title": "Lose Yourself", "artist": "Eminem" '

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text
    print(message)

    try:
        song_recommendations = json.loads(message)
        print("Recommended Songs:")
        for i, song in enumerate(song_recommendations, start=1):
            print(f"{i}. {song}")
        return song_recommendations
    except json.JSONDecodeError:
        print("Error decoding JSON response from OpenAI.")

def create_playlist(emotion_label):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                                   client_secret=CLIENT_SECRET,
                                                   redirect_uri=REDIRECT_URI,
                                                   scope='playlist-modify-public'))

    results = sp.current_user()
    user_id = results['id']
    playlist_name = f"TRY {emotion_label}"

    playlist = sp.user_playlist_create(f"{user_id}", f"TRY {emotion_label}", public=True)

    playlist_id = sp.current_user_playlists()['items'][0]['id']
    for song in song_recommendations:
        track_uri = sp.search(q=f"{song['title']} {song['artist']}", type='track')['tracks']['items'][0]['uri']
        sp.playlist_add_items(playlist_id, [track_uri])

    print(f"Playlist '{playlist_name}' created and songs added to it.")



emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded emotion detection model from disk")

cap = cv2.VideoCapture(0)

song_recommendations = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(num_faces) > 0:
        saved_frame = frame.copy()

        for (x, y, w, h) in num_faces[:1]:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            distribution = emotion_prediction[0]
            emotion_prediction = emotion_model.predict(cropped_img)
            max_emotion_score = max(emotion_prediction[0]) * 100
            max_emotion_index = np.argmax(emotion_prediction)
            emotion_label = emotion_dict[max_emotion_index]

            total_probability = np.sum(emotion_prediction)
            emotion_percentages = (emotion_prediction / total_probability) * 100

            for i, percentage in enumerate(emotion_percentages[0]):
                formatted_percentage = "{:.2f}".format(percentage)
                print(f"{emotion_dict[i]}: {formatted_percentage}")

            emotion_output = {emotion_dict[i].lower(): float(distribution[i]) for i in range(len(emotion_dict))}
            print(str(emotion_output))

            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        song_recommendations = music_recommendation(emotion_output)
        create_playlist(emotion_label)
        break

cv2.imshow('Emotion Detection', frame)

time.sleep(5)

cap.release()
cv2.destroyAllWindows()
