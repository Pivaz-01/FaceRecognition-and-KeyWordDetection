###########################################################################################################################
###########################################################################################################################
## ENVIRONMENT SETTINGS ##

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
from pyngrok import ngrok
import numpy as np
from sphereface_pytorch.net_sphere import sphere20a
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from base64 import b64decode
import dlib
import time
import torchaudio as ta
import tempfile
import os
from pyngrok.conf import PyngrokConfig
from flask import session

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngrok_config = PyngrokConfig(auth_token="2r1XXEr55mskib7Oy7zMemkYe6q_2dYqbYnCqKRFNXGfj8n3z")  

###########################################################################################################################
###########################################################################################################################
## MODELS DEFINITION ##

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("sphereface_pytorch/shape_predictor_68_face_landmarks.dat")

class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=112, desired_face_height=96):
        """
        Initialize the face aligner object
        :param desired_left_eye: Relative position of the left eye in the aligned image
        :param desired_face_width: Width of the output face image
        :param desired_face_height: Height of the output face image
        """
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

        # Calculate the desired right eye position based on the left eye position
        self.desired_right_eye_x = 1.0 - desired_left_eye[0]

        # Initialize the facial landmark predictor
        self.predictor = dlib.shape_predictor("sphereface_pytorch/shape_predictor_68_face_landmarks.dat")

    def align(self, image, face_rect):
        """
        Align the face in the image using facial landmarks
        :param image: Input image
        :param face_rect: dlib rectangle containing the face
        :return: Aligned face image
        """
        # Get facial landmarks
        landmarks = self.predictor(image, face_rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract the left and right eye coordinates
        left_eye = landmarks[36:42].mean(axis=0).astype("int")
        right_eye = landmarks[42:48].mean(axis=0).astype("int")

        # Calculate angle between eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate the desired right eye coordinate
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]

        # Calculate scaling factor
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.desired_face_width
        scale = desired_dist / dist

        # Calculate eyes center point
        eyes_center = (float(left_eye[0] + right_eye[0]) / 2,
               float(left_eye[1] + right_eye[1]) / 2)

        # Create rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update the translation component of the matrix
        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # Apply affine transformation
        aligned_face = cv2.warpAffine(image, M, (self.desired_face_width, self.desired_face_height),
                                    flags=cv2.INTER_CUBIC)

        return aligned_face

class WakeWordModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Mel Spectrogram transformation
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=64,
            power=2.0
        )

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 37, 256), # 8x37 is the spatial dimension of the feature map
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Convert to mel spectrogram
        x = self.mel_spectrogram(x)

        # Extract features
        x = self.features(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classify
        x = self.classifier(x)
        return x

face_model = sphere20a()
face_model.fc5 = torch.nn.Linear(face_model.fc5.in_features, 64)
face_model.fc6 = torch.nn.Linear(64, 3)
face_model.load_state_dict(torch.load("video_model.pth", map_location=device))
face_model = face_model
face_model.eval()

audio_model = WakeWordModel()
audio_model.load_state_dict(torch.load("audio_model.pth", map_location=device))
audio_model.eval()

face_aligner = FaceAligner()

###########################################################################################################################
###########################################################################################################################
## PROCESSING FUNCTIONS ##

def apply_noise_gate(waveform, threshold_db=-40):
    """
    Apply a noise gate to remove low volume sounds.
    Args:
        waveform (torch.Tensor): Input audio waveform
        threshold_db (float): Threshold in decibels below which audio will be silenced
    Returns:
        torch.Tensor: Processed waveform with noise gate applied
    """
    # Convert threshold from dB to amplitude
    threshold_amplitude = 10 ** (threshold_db / 20)

    # Calculate amplitude envelope
    window_size = 1024
    hop_length = 512
    envelope = torch.zeros_like(waveform)

    for i in range(0, waveform.size(1), hop_length):
        window = waveform[:, i:min(i + window_size, waveform.size(1))]
        envelope[:, i:i + window.size(1)] = torch.max(window.abs())

    # Create mask for samples above threshold
    mask = (envelope > threshold_amplitude).float()

    # Apply smooth fade to avoid jumps
    fade_samples = 128
    mask = F.pad(mask, (fade_samples, fade_samples))
    mask = F.avg_pool1d(mask, kernel_size=fade_samples*2 + 1, stride=1, padding=0)
    mask = mask[:, :waveform.size(1)]

    return waveform * mask

def center_audio(waveform, target_length, threshold_amplitude=0.15, sample_rate=16000):
    """
    Centers the audio around the region where the amplitude exceeds the threshold.
    It includes the region from the first sample that exceeds the threshold
    to the last sample that exceeds the threshold.

    Args:
        waveform (torch.Tensor): Input audio waveform (1 x N).
        target_length (int): Target length of the waveform (in samples).
        threshold_amplitude (float): Threshold for detecting amplitude peaks.
        sample_rate (int): Sample rate of the waveform (default 16 kHz).

    Returns:
        torch.Tensor: Centered waveform of target length.
    """
    # Find the indices of all samples where the amplitude exceeds the threshold
    exceed_idx = (waveform.abs() > threshold_amplitude).nonzero()

    if exceed_idx.size(0) == 0:
        raise ValueError("No samples exceed the threshold amplitude.")

    # Get the first and last index where the threshold is exceeded
    start_idx = exceed_idx[0, 1].item()  
    end_idx = exceed_idx[-1, 1].item()   

    # Define the segment that includes both the start and end indices
    audio_segment = waveform[:, start_idx:end_idx]

    # If the extracted segment is shorter than the target length, pad it
    if audio_segment.size(1) < target_length:
        pad_size = target_length - audio_segment.size(1)
        padding_left = pad_size // 2
        padding_right = pad_size - padding_left
        centered_waveform = F.pad(audio_segment, (padding_left, padding_right), "constant", 0)
    else:
        # If the extracted segment is longer, crop it to the target length
        start_crop = (audio_segment.size(1) - target_length) // 2
        centered_waveform = audio_segment[:, start_crop:start_crop + target_length]

    return centered_waveform

def process_audio(audio_bytes, target_sr=16000):
    """
    Process recorded audio to match model requirements with noise filtering,
    centering around highest volume, and applying advanced augmentations.
    """
    try:
        # Create a specific directory for temporary files
        temp_dir = os.path.join(os.path.expanduser("~"), "audio_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create temporary files for both raw and converted audio
        raw_temp_path = os.path.join(temp_dir, f"raw_audio_{time.time()}.webm")
        wav_temp_path = os.path.join(temp_dir, f"converted_audio_{time.time()}.wav")
        
        try:
            # Save raw audio data
            with open(raw_temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            # Convert to WAV using ffmpeg
            import subprocess
            command = [
                'ffmpeg',
                '-i', raw_temp_path,
                '-acodec', 'pcm_s16le',
                '-ar', str(target_sr),
                '-ac', '1',
                wav_temp_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            
            # Load the converted WAV file
            waveform, _ = ta.load(wav_temp_path)
            
            waveform = apply_noise_gate(waveform)
            waveform = waveform / (waveform.abs().max() + 1e-8)

            target_length = target_sr * 3  # 3 seconds
            waveform = center_audio(waveform, target_length)

            return waveform.unsqueeze(0)
            
        finally:
            # Clean up temporary files
            if os.path.exists(raw_temp_path):
                os.remove(raw_temp_path)
            if os.path.exists(wav_temp_path):
                os.remove(wav_temp_path)
                
    except Exception as e:
        raise

###########################################################################################################################
###########################################################################################################################
## FLASK APP ##

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>THE BOUNCER</title>
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            font-family: 'Open Sans', sans-serif; 
            font-weight: 1000; 
            background-color: #f4f4f4;
            flex-direction: column;
            position: relative;
        }

        #background-video {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: -1;
        }

        #content {
            display: flex;
            flex-direction: column;
            align-items: center;
            z-index: 1; 
        }

        @keyframes fadeIn {
            0% {
                opacity: 0;
            }
            100% {
                opacity: 1;
            }
        }

        @keyframes fadeOut {
            0% {
                opacity: 1;
            }
            100% {
                opacity: 0;
            }
        }

        #message {
            font-size: 3rem;
            text-align: center;
            color: #fff;
            font-weight: 1000;
            animation: fadeIn 1.5s ease-in-out; 
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            font-weight: 500; 
        }

        #video {
            display: none;
        }

        #reset-btn {
            margin-top: 20px;
            padding: 10px 20px;
            font-size: 1.2rem;
            font-weight: 500; 
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            display: none; 
            animation: fadeIn 1.5s ease-in-out;
        }

        #reset-btn:hover {
            background-color: #45a049;
        }

    </style>
</head>
<body>
    <video id="background-video" autoplay loop muted>
        <source src="/static/background.mp4" type="video/mp4">
    </video>

    <div id="content">
        <video id="video" autoplay></video>
        <h1 id="message">Stay there...</h1>
        <button id="reset-btn" onclick="resetApp()">Restart</button>
    </div>

    <script>
        let video = document.getElementById('video');
        let mediaStream = null;

        async function startVideo() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    headers: {
                        'ngrok-skip-browser-warning': 'true'
                    }
                });
                video.srcObject = mediaStream;
                sendVideoFrames();
            } catch (err) {
                console.error("Error:", err);
            }
        }

        async function sendVideoFrames() {
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            const ctx = canvas.getContext('2d');

            async function capture() {
                if (!mediaStream) return;

                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);

                try {
                    const response = await fetch('/process_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData })
                    });

                    const result = await response.json();
                    const messageElement = document.getElementById('message');
                    messageElement.innerText = result.message;

                    if (result.status === 'no_face_detected') {
                        messageElement.style.color = 'white';
                        document.getElementById('reset-btn').style.display = 'block';  // Add this line
                        return;  
                    } else {
                        messageElement.style.color = 'white';
                    }

                    if (result.proceed_to_audio) {
                        stopVideo();
                        startAudioRecording();
                        return;
                    }

                    requestAnimationFrame(capture);
                } catch (err) {
                    console.error("Error:", err);
                    requestAnimationFrame(capture);
                }
            }

            capture();
        }

        function stopVideo() {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
            }
            video.srcObject = null;
        }

        async function startAudioRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const mediaRecorder = new MediaRecorder(stream);
                const audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks);
                    const reader = new FileReader();

                    reader.onloadend = async () => {
                        const base64Audio = reader.result.split(',')[1];
                        const response = await fetch('/process_audio', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({audio: base64Audio})
                        });

                        const result = await response.json();
                        document.getElementById('message').innerText = result.message;
                        document.getElementById('reset-btn').style.display = 'block';  

                    };

                    reader.readAsDataURL(audioBlob);
                };

                mediaRecorder.start();
                setTimeout(() => mediaRecorder.stop(), 3000);

            } catch (err) {
                console.error("Error:", err);
            }
        }

        function resetApp() {
            stopVideo();

            const message = document.getElementById('message');
            message.style.animation = 'fadeOut 1s ease-in-out forwards'; 

            setTimeout(() => {
                message.style.animation = 'fadeIn 1.5s ease-in-out'; 
                message.innerText = 'Stay there...';
            }, 1000); // Aspetta che l'animazione di uscita finisca prima di far tornare il messaggio

            startVideo();

            document.getElementById('reset-btn').style.display = 'none';
        }

        startVideo();
    </script>
</body>
</html>
"""

temp_dir = tempfile.gettempdir()

app = Flask(__name__)
app.secret_key = 'coccodrillo'  

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    img_data = request.json['image']

    # Convert base64 to image
    img_bytes = b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect faces
    faces, scores, _ = face_detector.run(img, 1, -1)

    score = max(scores) if len(scores) > 0 else 0

    if score >= .3:
        # Initialize or get face-related session variables
        face_detections = session.get('face_detections', 0)

        # Reset no_face_start if a face is detected
        session['no_face_start'] = None

        # Increment counter for consecutive face detections
        face_detections += 1
        session['face_detections'] = face_detections

        # Proceed to face recognition after 5 consecutive face detections
        if face_detections >= 5:
            # Face recognition logic (same as before)
            face_idx = np.argmax(scores)
            face_rect = faces[face_idx]

            aligned_face = face_aligner.align(img, face_rect)
            if aligned_face is None:
                session['consecutive_recognitions'] = 0
                return jsonify({
                    'status': 'processing',
                    'message': 'Stay there...',
                    'proceed_to_audio': False
                })

            processed_image = cv2.resize(aligned_face, (112, 96))
            processed_image = aligned_face / 255.0
            processed_image = torch.tensor(processed_image, dtype=torch.float32).unsqueeze(0)
            processed_image = processed_image.permute(0, 3, 1, 2).to(device)
            processed_image = processed_image.contiguous()

            with torch.no_grad():
                prob = F.softmax(face_model(processed_image), dim=1)
                _, preds = torch.max(prob, 1)
                result = preds.item()

            consecutive_recognitions = session.get('consecutive_recognitions', 0)
            last_recognized_face = session.get('last_recognized_face', None)

            if last_recognized_face != result:
                consecutive_recognitions = 0
            else:
                consecutive_recognitions += 1

            session['consecutive_recognitions'] = consecutive_recognitions
            session['last_recognized_face'] = result

            face_result = {
                1: "Hello Pivaz!",
                2: "Hello Martin!",
                0: "Unknown person detected"
            }

            if consecutive_recognitions >= 5:
                session['face_result'] = result
                return jsonify({
                    'status': 'face_detected',
                    'message': f"{face_result[result]}\nWhere do you want to go?",
                    'proceed_to_audio': True
                })

        return jsonify({
            'status': 'processing',
            'message': 'Stay there...',
            'proceed_to_audio': False
        })
    
    else: 
        empty_faces = session.get('empty_faces', 0)
        empty_faces += 1

        if empty_faces >= 50:
            session['face_detections'] = 0
            session['empty_faces'] = 0
            return jsonify({
                'status': 'no_face_detected',
                'message': 'No face detected',
                'proceed_to_audio': False
            })
        
        else:
            session['empty_faces'] = empty_faces
            return jsonify({
                'status': 'processing',
                'message': 'Stay there...',
                'proceed_to_audio': False
            })

@app.route('/process_audio', methods=['POST'])
def process_audio_endpoint():
    face_result = session.get('face_result') 
    audio_data = request.json['audio']
    audio_bytes = b64decode(audio_data)

    try:
        # Process the audio using the new methods
        waveform = process_audio(audio_bytes)
        waveform = waveform.to(device)

        # Make prediction
        with torch.no_grad():
            output = audio_model(waveform)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities)

        # Get probabilities for each class
        prob_dict = {
            "No wake word": f"{probabilities[0][0].item()*100:.2f}%",
            "Jarvis": f"{probabilities[0][1].item()*100:.2f}%",
            "Snowboy": f"{probabilities[0][2].item()*100:.2f}%"
        }

        # Check of accesses
        access_message = ""
        if face_result == 1:  
            if predicted_class.item() == 2:  
                access_message = "Section Snowboy: access granted!"
            elif predicted_class.item() == 1:  
                access_message = "Section Jarvis: access granted!"
            else:
                access_message = "Section not recognized, please restart"
        elif face_result == 2:  
            if predicted_class.item() == 2:  
                access_message = "Section Snowboy: access granted!"
            elif predicted_class.item() == 1:  
                access_message = "Section Jarvis: access granted!"
            else:
                access_message = "Section not recognized, please try again!"
        else:
            if predicted_class.item() == 1:  
                access_message = "Section Jarvis: access granted!"
            elif predicted_class.item() == 2: 
                access_message = "Section Snowboy: access denied!"
            else:
                access_message = "Section not recognized, please try again!"

        return jsonify({
            'message': f"{access_message}",
            'probabilities': prob_dict
        })

    except Exception as e:
        return jsonify({
            'message': "No audio detected, please restart",
            'error': str(e)
        })
    
###########################################################################################################################
###########################################################################################################################
## NGROK TUNNEL ##

ngrok_url = ngrok.connect(5000, bind_tls=True, pyngrok_config=ngrok_config)
public_url = f"{ngrok_url.public_url}?ngrok-skip-browser-warning=true"
print(f"Access URL: {public_url}")

app.run()
