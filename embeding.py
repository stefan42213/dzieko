import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        embedding = resnet(face)
        return embedding.detach().cpu().numpy()[0]
    return None

database = {}
base_folder = 'Trening'

for img_name in os.listdir(base_folder):
    if img_name.lower().endswith('.jpg'):
        name = os.path.splitext(img_name)[0]
        img_path = os.path.join(base_folder, img_name)
        emb = get_embedding(img_path)
        if emb is not None:
            database[name] = emb
            print(f'Zapisano embedding dla: {name}')

np.save('face_database.npy', database)
