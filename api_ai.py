import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
import io
import os
import logging
import traceback

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Dodaj CORS, aby frontend mógł się łączyć z innego adresu/IP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pozwól na wszystkie domeny
    allow_credentials=True,
    allow_methods=["*"],  # Pozwól na wszystkie metody
    allow_headers=["*"],  # Pozwól na wszystkie nagłówki
)

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Wczytaj bazę embeddingów
database = {}
if os.path.exists('face_database.npy'):
    try:
        database = np.load('face_database.npy', allow_pickle=True).item()
        logger.info(f"Wczytano bazę danych z {len(database)} twarzy")
    except Exception as e:
        logger.error(f"Błąd wczytywania bazy danych: {e}")
        database = {}
else:
    logger.warning("Brak pliku face_database.npy!")

# Inicjalizacja modeli z obsługą błędów
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Używane urządzenie: {device}")
    
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    logger.info("Modele załadowane pomyślnie")
except Exception as e:
    logger.error(f"Błąd ładowania modeli: {e}")
    raise

def recognize_face_from_bytes(image_bytes):
    try:
        # Otwórz obraz
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info(f"Załadowano obraz o rozmiarze: {image.size}")
        
        # Wykryj twarz
        face = mtcnn(image)
        if face is not None:
            logger.info("Wykryto twarz")
            face = face.unsqueeze(0).to(device)
            
            # Wygeneruj embedding
            with torch.no_grad():  # Dodaj to dla lepszej wydajności
                embedding = resnet(face).detach().cpu().numpy()
            
            # Znajdź najlepsze dopasowanie
            min_dist = float('inf')
            identity = 'Nieznana'
            
            if len(database) == 0:
                logger.warning("Baza danych jest pusta!")
                return 'Baza danych pusta', None
            
            for name, db_emb in database.items():
                dist = np.linalg.norm(embedding - db_emb)
                if dist < min_dist and dist < 1.0:  # Próg rozpoznania
                    min_dist = dist
                    identity = name
            
            logger.info(f"Rozpoznano jako: {identity} (dystans: {min_dist})")
            return identity, float(min_dist)
        else:
            logger.warning("Nie wykryto twarzy na obrazie")
            return 'Brak twarzy', None
            
    except Exception as e:
        logger.error(f"Błąd w rozpoznawaniu twarzy: {e}")
        logger.error(traceback.format_exc())
        return 'Błąd przetwarzania', None

@app.get("/")
async def root():
    return {
        "message": "API do rozpoznawania twarzy działa",
        "endpoints": ["/recognize/", "/health", "/docs"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_loaded": len(database) > 0,
        "database_size": len(database)
    }

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    try:
        # Sprawdź typ pliku
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Plik musi być obrazem")
        
        # Przeczytaj bytes
        img_bytes = await file.read()
        logger.info(f"Otrzymano plik: {file.filename}, rozmiar: {len(img_bytes)} bajtów")
        
        # Rozpoznaj twarz
        identity, distance = recognize_face_from_bytes(img_bytes)
        
        return {
            "identity": identity,
            "distance": distance,
            "filename": file.filename,
            "file_size": len(img_bytes)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd w endpoint /recognize/: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")

# Dodatkowy endpoint do debugowania
@app.post("/debug/")
async def debug_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Sprawdź czy MTCNN wykrywa twarz
        face = mtcnn(image)
        
        return {
            "filename": file.filename,
            "image_size": image.size,
            "image_mode": image.mode,
            "face_detected": face is not None,
            "file_size": len(img_bytes),
            "device": device
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd debugowania: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)