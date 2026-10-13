"""FacialNet web UI: a local server for live and single-image emotion recognition, and GAN face generation.

Uses only the Python standard library on top of the project's own requirements
(torch, torchvision, OpenCV, Pillow, NumPy), so no extra packages are needed.

    cd FacialNet
    .venv/bin/python webui/server.py            # then open http://localhost:8000
    .venv/bin/python webui/server.py --port 9000 --host 0.0.0.0
"""

import argparse
import base64
import io
import json
import mimetypes
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
VIDEO_DIR = ROOT / "Video Predicition"
DNCNN_PATH = ROOT / "Denoising" / "dncnn_rafdb_finetuned.pth"
RESULTS_PATH = ROOT / "Denoising" / "results.json"
GAN_PATH = ROOT / "GAN" / "cgan_rafdb_generator.pth"
GAN_RESULTS_PATH = ROOT / "GAN" / "results.json"
RAFDB_TEST = ROOT / "data" / "raf-db" / "test"

# RAF-DB folder order 1-7 (ImageFolder sorts the folder names)
CLASS_NAMES = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]

ARCHS = {
    "efficientnet": {"label": "EfficientNet-B0", "file": "efficientnet_b0_rafdb_final.pth"},
    "densenet": {"label": "DenseNet-121", "file": "densenet121_rafdb_final.pth"},
    "resnet": {"label": "ResNet-18", "file": "resnet18_rafdb_final.pth"},
}

WEIGHT_SETS = {
    "current": {"label": "Current (20 epochs)", "dir": VIDEO_DIR},
    "aug12": {"label": "Full aug, 12 epochs", "dir": VIDEO_DIR / "aug12_weights"},
    "baseline": {"label": "Flip + rotation, 12 epochs", "dir": VIDEO_DIR / "baseline_weights"},
}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# One model runs at a time; the GPU is shared and the Haar cascade is not thread safe
INFER_LOCK = threading.Lock()
_models = {}
_denoiser = None
_generator = None


class DnCNN(nn.Module):
    """DnCNN denoiser from Denoising/denoising.ipynb: predicts the noise map and subtracts it."""

    def __init__(self, channels=3, depth=20, features=64):
        super().__init__()
        layers = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features, channels, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, noisy):
        return noisy - self.model(noisy)


class ConditionalBatchNorm2d(nn.Module):
    """BatchNorm with a learned scale and shift per emotion. y holds class weights (one-hot, or a mix of emotions)."""

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gain = nn.Embedding(num_classes, num_features)
        self.bias = nn.Embedding(num_classes, num_features)

    def forward(self, x, y):
        return self.bn(x) * (y @ self.gain.weight)[:, :, None, None] + (y @ self.bias.weight)[:, :, None, None]


class Generator(nn.Module):
    """Conditional DCGAN generator from GAN/gan.ipynb: 64x64 faces in [-1, 1]."""

    z_dim = 128

    def __init__(self, num_classes=7, channels=(512, 256, 128, 64)):
        super().__init__()
        self.project = nn.ConvTranspose2d(self.z_dim, channels[0], 4, 1, 0, bias=False)
        self.bns = nn.ModuleList(ConditionalBatchNorm2d(c, num_classes) for c in channels)
        self.ups = nn.ModuleList(nn.ConvTranspose2d(channels[i], channels[i + 1], 4, 2, 1, bias=False)
                                 for i in range(len(channels) - 1))
        self.out = nn.ConvTranspose2d(channels[-1], 3, 4, 2, 1)

    def forward(self, z, y):
        h = self.project(z.view(z.size(0), -1, 1, 1))
        for bn, up in zip(self.bns, self.ups):
            h = up(F.relu(bn(h, y)))
        return torch.tanh(self.out(F.relu(self.bns[-1](h, y))))


def build_model(arch):
    if arch == "efficientnet":
        model = models.efficientnet_b0()
        model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.classifier[1].in_features, 7))
    elif arch == "densenet":
        model = models.densenet121()
        model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.classifier.in_features, 7))
    else:
        model = models.resnet18()
        model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 7))
    return model


def weights_path(arch, weight_set):
    return WEIGHT_SETS[weight_set]["dir"] / ARCHS[arch]["file"]


def get_model(arch, weight_set):
    key = (arch, weight_set)
    if key not in _models:
        model = build_model(arch)
        model.load_state_dict(torch.load(weights_path(arch, weight_set), map_location="cpu"))
        _models[key] = model.to(DEVICE).eval()
    return _models[key]


def get_denoiser():
    global _denoiser
    if _denoiser is None:
        denoiser = DnCNN()
        denoiser.load_state_dict(torch.load(DNCNN_PATH, map_location="cpu"))
        _denoiser = denoiser.to(DEVICE).eval()
    return _denoiser


def get_generator():
    global _generator
    if _generator is None:
        generator = Generator(len(CLASS_NAMES))
        generator.load_state_dict(torch.load(GAN_PATH, map_location="cpu"))
        _generator = generator.to(DEVICE).eval()
    return _generator


def to_data_url(pil_img, size=160):
    img = pil_img.copy()
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def detect_faces(frame_bgr, min_size):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(min_size, min_size))
    return [tuple(int(v) for v in f) for f in faces]


def pad_box(box, shape, pad):
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(shape[1], x + w + px), min(shape[0], y + h + py)
    return x0, y0, x1 - x0, y1 - y0


@torch.no_grad()
def classify_faces(frame_rgb, boxes, archs, weight_set, noise_sigma, denoise, return_crops, rng):
    """Classify each face box with every requested model. Noise and denoising are applied per face,
    as in the Denoising study: y = clip(x + n, 0, 1), n ~ N(0, sigma^2) on the 0-255 scale."""
    faces = []
    for box in boxes:
        x, y, w, h = box
        crop = frame_rgb[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        face = torch.from_numpy(crop).permute(2, 0, 1).float().div(255).unsqueeze(0)
        stages = {}
        if return_crops:
            stages["input"] = to_data_url(Image.fromarray(crop))
        if noise_sigma > 0:
            noise = torch.from_numpy(rng.normal(0, noise_sigma / 255, face.shape).astype(np.float32))
            face = (face + noise).clamp(0, 1)
            if return_crops:
                stages["noisy"] = to_data_url(transforms.functional.to_pil_image(face[0]))
        if denoise:
            face = get_denoiser()(face.to(DEVICE)).clamp(0, 1).cpu()
            if return_crops:
                stages["denoised"] = to_data_url(transforms.functional.to_pil_image(face[0]))

        pil_face = transforms.functional.to_pil_image(face[0])
        batch = PREPROCESS(pil_face).unsqueeze(0).to(DEVICE)
        results = {}
        for arch in archs:
            probs = F.softmax(get_model(arch, weight_set)(batch), dim=1)[0].cpu().tolist()
            top = int(np.argmax(probs))
            results[arch] = {"probs": probs, "top": top, "label": CLASS_NAMES[top], "confidence": probs[top]}
        faces.append({"box": list(box), "results": results, "stages": stages})
    return faces


def predict(image_bytes, params):
    archs = [a for a in params.get("models", "efficientnet").split(",") if a in ARCHS] or ["efficientnet"]
    weight_set = params.get("weights", "current")
    if weight_set not in WEIGHT_SETS:
        weight_set = "current"
    detect = params.get("detect", "auto")  # auto | on | off
    pad = min(max(float(params.get("pad", 0.05)), 0.0), 0.5)
    noise_sigma = min(max(float(params.get("noise", 0)), 0.0), 100.0)
    denoise = params.get("denoise") in ("1", "true")
    return_crops = params.get("crops") in ("1", "true")
    max_faces = int(params.get("max_faces", 6))

    if denoise and not DNCNN_PATH.exists():
        raise ValueError("DnCNN weights not found at Denoising/dncnn_rafdb_finetuned.pth")

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the image")
    h, w = frame.shape[:2]
    rng = np.random.default_rng()

    t0 = time.perf_counter()
    with INFER_LOCK:
        boxes, mode = [], "whole"
        if detect != "off":
            raw = detect_faces(frame, max(24, min(h, w) // 12))
            raw.sort(key=lambda b: b[2] * b[3], reverse=True)
            boxes = [pad_box(b, frame.shape, pad) for b in raw[:max_faces]]
            mode = "detected"
        # Aligned face crops such as RAF-DB are often too tight for the Haar cascade
        if not boxes and detect != "on":
            boxes, mode = [(0, 0, w, h)], "whole"
        t1 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = classify_faces(frame_rgb, boxes, archs, weight_set, noise_sigma, denoise, return_crops, rng)
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        t2 = time.perf_counter()

    return {
        "width": w,
        "height": h,
        "mode": mode,
        "faces": faces,
        "timing": {"detect_ms": (t1 - t0) * 1000, "infer_ms": (t2 - t1) * 1000},
    }


def sample_image(label=None):
    folders = [d for d in sorted(RAFDB_TEST.iterdir()) if d.is_dir()] if RAFDB_TEST.exists() else []
    if not folders:
        raise ValueError("RAF-DB test set not found in data/raf-db/test")
    if label is not None and 0 <= label < len(folders):
        folder = folders[label]
    else:
        folder = random.choice(folders)
    path = random.choice([p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    data = path.read_bytes()
    return {
        "name": path.name,
        "label": folders.index(folder),
        "image": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
    }


def generate(params):
    """Generate faces with the conditional GAN and classify each one with a RAF-DB model.

    mode=label:    n faces, all of one emotion (label=0-6) or cycling through the seven (label=all)
    mode=emotions: one random face rendered with each of the seven emotions
    mode=morph:    one random face morphing from emotion `from` to emotion `to` in `steps` steps
    The same seed always gives the same random faces, so changing the emotion keeps the people.
    """
    if not GAN_PATH.exists():
        raise ValueError("GAN weights not found at GAN/cgan_rafdb_generator.pth")
    arch = params.get("model", "efficientnet")
    if arch not in ARCHS:
        arch = "efficientnet"
    weight_set = params.get("weights", "current")
    if weight_set not in WEIGHT_SETS:
        weight_set = "current"
    mode = params.get("mode", "label")
    seed = int(params["seed"]) if params.get("seed", "").isdigit() else random.randrange(1_000_000)
    num_classes = len(CLASS_NAMES)

    if mode == "emotions":
        y = torch.eye(num_classes)
        z_count = 1
    elif mode == "morph":
        a = min(max(int(params.get("from", 6)), 0), num_classes - 1)
        b = min(max(int(params.get("to", 3)), 0), num_classes - 1)
        t = torch.linspace(0, 1, min(max(int(params.get("steps", 9)), 2), 24))
        y = torch.zeros(len(t), num_classes)
        y[:, a] += 1 - t
        y[:, b] += t
        z_count = 1
    else:
        n = min(max(int(params.get("n", 8)), 1), 48)
        label = params.get("label", "all")
        labels = [int(label)] * n if label.isdigit() and int(label) < num_classes else [i % num_classes for i in range(n)]
        y = F.one_hot(torch.tensor(labels), num_classes).float()
        z_count = n

    z = torch.randn(z_count, Generator.z_dim, generator=torch.Generator().manual_seed(seed))
    z = z.expand(len(y), -1) if z_count == 1 else z

    t0 = time.perf_counter()
    with INFER_LOCK, torch.no_grad():
        faces = (get_generator()(z.to(DEVICE), y.to(DEVICE)) * 0.5 + 0.5).clamp(0, 1)
        t1 = time.perf_counter()
        batch = F.interpolate(faces, size=(224, 224), mode="bilinear", align_corners=False)
        probs = F.softmax(get_model(arch, weight_set)((batch - IMAGENET_MEAN) / IMAGENET_STD), dim=1).cpu()
        faces = faces.cpu()
        t2 = time.perf_counter()

    out = []
    for face, mix, p in zip(faces, y.tolist(), probs.tolist()):
        top = int(np.argmax(p))
        img = transforms.functional.to_pil_image(face).resize((128, 128), Image.LANCZOS)
        out.append({
            "image": to_data_url(img, size=128),
            "mix": mix,
            "target": int(np.argmax(mix)),
            "probs": p,
            "top": top,
            "label": CLASS_NAMES[top],
            "confidence": p[top],
        })
    return {
        "seed": seed,
        "mode": mode,
        "model": arch,
        "faces": out,
        "timing": {"generate_ms": (t1 - t0) * 1000, "classify_ms": (t2 - t1) * 1000},
    }


def info():
    available = {}
    for ws in WEIGHT_SETS:
        available[ws] = [a for a in ARCHS if weights_path(a, ws).exists()]
    results = None
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
    gan_results = None
    if GAN_RESULTS_PATH.exists():
        gan_results = json.loads(GAN_RESULTS_PATH.read_text())
        gan_results.pop("history", None)
    return {
        "device": DEVICE.type,
        "classes": CLASS_NAMES,
        "archs": {k: v["label"] for k, v in ARCHS.items()},
        "weight_sets": {k: v["label"] for k, v in WEIGHT_SETS.items()},
        "available": available,
        "denoiser": DNCNN_PATH.exists(),
        "samples": RAFDB_TEST.exists(),
        "results": results,
        "gan": GAN_PATH.exists(),
        "gan_results": gan_results,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "FacialNet"

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/predict"):
            super().log_message(fmt, *args)

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/info":
                return self.send_json(info())
            if url.path == "/api/sample":
                label = int(params["label"]) if params.get("label", "").isdigit() else None
                return self.send_json(sample_image(label))
            if url.path == "/api/generate":
                return self.send_json(generate(params))
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 400)

        rel = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
        path = (STATIC / rel).resolve()
        if not path.is_file() or STATIC not in path.parents:
            return self.send_json({"error": "not found"}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/predict":
            return self.send_json({"error": "not found"}, 404)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 25 * 1024 * 1024:
            return self.send_json({"error": "Send an image of at most 25 MB as the request body"}, 400)
        try:
            return self.send_json(predict(self.rfile.read(length), params))
        except (ValueError, FileNotFoundError) as exc:
            return self.send_json({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description="FacialNet web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading EfficientNet-B0 on {DEVICE}...")
    get_model("efficientnet", "current")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FacialNet web UI running at http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
