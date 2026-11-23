"""poultry_core.py -- shared inference core for the Poultry Fecal-Image Screening system.

Used by BOTH the training notebook (for gate calibration, Grad-CAM and QA) and the
deployed web app, so training-time and serving-time preprocessing cannot drift apart.

Pipeline for one uploaded image
  1. Gate A (optional): CLIP zero-shot "is this poultry droppings?"   (needs NO extra dataset)
  2. Classifier: 4-class transfer-learning CNN (Coccidiosis/Healthy/Newcastle/Salmonellosis)
  3. Gate B: Mahalanobis distance in the CNN feature space (fitted on poultry data only)
  4. Gate C: softmax-confidence rejection
Research prototype only -- not a substitute for laboratory diagnosis.
"""
from __future__ import annotations

import json
import os
import threading
import warnings

import numpy as np
from PIL import Image, ImageOps

warnings.filterwarnings("ignore", message="The structure of `inputs`")

SOURCE_SIZE = 256  # the training images were distributed pre-resized to 256x256
DEFAULT_CLASS_NAMES = ["Coccidiosis", "Healthy", "Newcastle_Disease", "Salmonellosis"]
DISCLAIMER = (
    "Research prototype for decision support only. It does not replace laboratory "
    "confirmation (PCR, culture, oocyst count) or a veterinarian's judgement."
)

VERDICT_PREDICTION = "PREDICTION"
VERDICT_UNCERTAIN = "UNCERTAIN"
VERDICT_NOT_POULTRY = "NOT_POULTRY_FECES"
VERDICT_OOD = "OUT_OF_DISTRIBUTION"

# ----------------------------------------------------------------------------------
# CLIP zero-shot prompt bank (no training data needed)
# ----------------------------------------------------------------------------------
CLIP_POULTRY = [
    "chicken droppings", "poultry feces on a farm floor",
    "chicken manure", "fresh droppings in a chicken coop",
]
CLIP_OTHER_FECES = [
    "cow dung", "dog feces", "cat feces", "horse manure", "sheep droppings",
    "goat droppings", "pig manure", "human stool", "rabbit droppings",
    "elephant dung", "wild bird droppings on a pavement",
]
CLIP_NOT_FECES = [
    "a live chicken", "a person", "food on a plate", "a landscape", "a building",
    "soil and mud", "a plant with leaves", "a screenshot of text", "a car", "a pet animal",
]


def clip_prompts():
    out = [("poultry", f"a photo of {p}") for p in CLIP_POULTRY]
    out += [("other_feces", f"a photo of {p}") for p in CLIP_OTHER_FECES]
    out += [("not_feces", f"a photo of {p}") for p in CLIP_NOT_FECES]
    return out


# ----------------------------------------------------------------------------------
# Preprocessing (must match the training pipeline exactly)
# ----------------------------------------------------------------------------------
def get_preprocess(arch: str):
    import keras
    table = {
        "MobileNetV2": keras.applications.mobilenet_v2.preprocess_input,
        "DenseNet169": keras.applications.densenet.preprocess_input,
        "EfficientNetB0": keras.applications.efficientnet.preprocess_input,
        "ResNet50": keras.applications.resnet50.preprocess_input,
    }
    return table[arch]


def load_rgb(img) -> Image.Image:
    """Accepts a path, PIL image or numpy array; applies EXIF rotation; returns RGB PIL."""
    if isinstance(img, (str, os.PathLike)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8"))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def to_model_array(img, size=(224, 224)) -> np.ndarray:
    """Phone photo / dataset image -> float32 array in the 0..255 range, shape (H, W, 3).

    Mirrors training: (dataset is 256x256) -> tf.image.resize bilinear -> round to uint8.
    """
    import tensorflow as tf
    img = load_rgb(img).resize((SOURCE_SIZE, SOURCE_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    out = tf.image.resize(arr, size, method="bilinear").numpy()
    return np.clip(np.round(out), 0, 255).astype(np.float32)


# ----------------------------------------------------------------------------------
# Threshold helpers (calibrated on IN-distribution data only)
# ----------------------------------------------------------------------------------
def threshold_accept_high(scores_in, accept_rate):
    """Accept if score >= t, so that `accept_rate` of in-distribution samples pass."""
    return float(np.quantile(np.asarray(scores_in, dtype=np.float64), 1.0 - accept_rate))


def threshold_accept_low(scores_in, accept_rate):
    """Accept if score <= t, so that `accept_rate` of in-distribution samples pass."""
    return float(np.quantile(np.asarray(scores_in, dtype=np.float64), accept_rate))


# ----------------------------------------------------------------------------------
# Gate B: Mahalanobis distance on penultimate CNN features
# ----------------------------------------------------------------------------------
class MahalanobisGate:
    """Class-conditional Gaussians with a shared (Ledoit-Wolf shrunk) covariance."""

    def __init__(self, means: np.ndarray, precision: np.ndarray):
        self.means = np.asarray(means, dtype=np.float64)
        self.precision = np.asarray(precision, dtype=np.float64)

    @classmethod
    def fit(cls, feats, labels, num_classes):
        from sklearn.covariance import LedoitWolf
        feats = np.asarray(feats, dtype=np.float64)
        labels = np.asarray(labels)
        means = np.stack([feats[labels == c].mean(axis=0) for c in range(num_classes)])
        centered = feats - means[labels]
        precision = LedoitWolf().fit(centered).precision_
        return cls(means, precision)

    def score(self, feats) -> np.ndarray:
        """Minimum (over classes) squared Mahalanobis distance / D.  Larger = more unfamiliar."""
        feats = np.asarray(feats, dtype=np.float64)
        d = feats.shape[1]
        best = None
        for mu in self.means:
            diff = feats - mu
            dist = ((diff @ self.precision) * diff).sum(axis=1) / d
            best = dist if best is None else np.minimum(best, dist)
        return best

    def save(self, path):
        np.savez_compressed(path, means=self.means.astype(np.float32),
                            precision=self.precision.astype(np.float32))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["means"], z["precision"])


# ----------------------------------------------------------------------------------
# Gate A: CLIP zero-shot "poultry droppings vs everything else"
# ----------------------------------------------------------------------------------
class ClipGate:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        prompts = clip_prompts()
        self.groups = np.array([g for g, _ in prompts])
        self.texts = [t for _, t in prompts]
        tok = self.processor.tokenizer(self.texts, padding=True, return_tensors="pt")
        self.input_ids = tok["input_ids"].to(self.device)
        self.attention_mask = tok["attention_mask"].to(self.device)
        self.poultry_mask = self.groups == "poultry"

    def score(self, images, batch_size=32):
        """Returns (p_poultry[N], top_prompt[N]).  p_poultry = softmax mass on poultry prompts."""
        p_out, top_out = [], []
        for i in range(0, len(images), batch_size):
            batch = [load_rgb(x) for x in images[i:i + batch_size]]
            pix = self.processor.image_processor(batch, return_tensors="pt")["pixel_values"]
            with self.torch.no_grad():
                logits = self.model(input_ids=self.input_ids,
                                    attention_mask=self.attention_mask,
                                    pixel_values=pix.to(self.device)).logits_per_image
                probs = logits.softmax(dim=-1).cpu().numpy()
            p_out.extend(probs[:, self.poultry_mask].sum(axis=1).tolist())
            top_out.extend(self.texts[j].replace("a photo of ", "") for j in probs.argmax(axis=1))
        return np.array(p_out), top_out


# ----------------------------------------------------------------------------------
# Grad-CAM (works with the nested-backbone model built in the notebook)
# ----------------------------------------------------------------------------------
def gradcam_heatmap(model, x, class_index=None):
    """x: preprocessed batch (1,H,W,3). Uses the backbone's final feature map."""
    import keras
    import tensorflow as tf
    base = next((l for l in model.layers if isinstance(l, keras.Model)), None)
    if base is None:
        raise ValueError("No nested backbone found for Grad-CAM.")
    head = model.layers[model.layers.index(base) + 1:]
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        fmap = base(x, training=False)
        tape.watch(fmap)
        h = fmap
        for layer in head:
            try:
                h = layer(h, training=False)
            except TypeError:
                h = layer(h)
        if class_index is None:
            class_index = int(tf.argmax(h[0]))
        target = h[:, class_index]
    grads = tape.gradient(target, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), class_index


def _jet(v):
    v = np.clip(v, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_cam(pil_img: Image.Image, cam: np.ndarray, alpha=0.45, out_size=(448, 448)) -> Image.Image:
    base = load_rgb(pil_img).resize(out_size, Image.BILINEAR)
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(out_size, Image.BICUBIC)
    heat = _jet(np.asarray(cam_img, dtype=np.float32) / 255.0) * 255.0
    blend = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * heat
    return Image.fromarray(np.clip(blend, 0, 255).astype("uint8"))


# ----------------------------------------------------------------------------------
# The analyzer used by the web app and by the notebook QA cells
# ----------------------------------------------------------------------------------
class PoultryAnalyzer:
    def __init__(self, bundle_dir, use_clip=None):
        import keras
        self.bundle_dir = bundle_dir
        with open(os.path.join(bundle_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.class_names = self.meta.get("class_names", DEFAULT_CLASS_NAMES)
        self.arch = self.meta["arch"]
        self.size = tuple(self.meta.get("image_size", [224, 224]))
        self.conf_thr = float(self.meta.get("conf_threshold", 0.75))
        self.preprocess = get_preprocess(self.arch)
        self.model = keras.models.load_model(os.path.join(bundle_dir, "model.keras"), compile=False)
        feat_out = self.model.get_layer(self.meta.get("feature_layer", "gap")).output
        self.multi = keras.Model(self.model.inputs, [feat_out, self.model.output])
        self.lock = threading.Lock()
        self.warnings = []

        self.maha, self.maha_thr = None, None
        mpath = os.path.join(bundle_dir, "maha.npz")
        if os.path.isfile(mpath) and "maha" in self.meta:
            self.maha = MahalanobisGate.load(mpath)
            self.maha_thr = float(self.meta["maha"]["threshold"])

        self.clip, self.clip_thr = None, None
        cmeta = self.meta.get("clip", {})
        want = cmeta.get("enabled", False) if use_clip is None else use_clip
        if os.environ.get("USE_CLIP") == "0":
            want = False
        if want:
            try:
                self.clip = ClipGate(cmeta.get("model_name", "openai/clip-vit-base-patch32"))
                self.clip_thr = float(cmeta["threshold"])
            except Exception as e:  # noqa: BLE001 - degrade gracefully
                self.warnings.append(
                    f"CLIP gate unavailable ({type(e).__name__}). Non-poultry rejection "
                    "falls back to feature-space outlier detection only.")

    # -- batch scoring -------------------------------------------------------------
    def _prepare(self, images):
        arrs = [to_model_array(im, self.size) for im in images]
        return np.stack([self.preprocess(a.copy()) for a in arrs]).astype("float32")

    def score_batch(self, images, batch_size=32):
        """Raw scores for a list of images (paths / PIL / arrays)."""
        probs, feats = [], []
        for i in range(0, len(images), batch_size):
            x = self._prepare(images[i:i + batch_size])
            with self.lock:
                f, p = self.multi(x, training=False)
            feats.append(np.asarray(f))
            probs.append(np.asarray(p))
        probs, feats = np.concatenate(probs), np.concatenate(feats)
        out = {"probs": probs,
               "maha": self.maha.score(feats) if self.maha is not None else None,
               "clip_p": None, "clip_top": None}
        if self.clip is not None:
            out["clip_p"], out["clip_top"] = self.clip.score(images)
        return out

    # -- decision logic ------------------------------------------------------------
    def decide(self, probs_row, maha_s=None, clip_p=None, clip_top=None):
        top = int(np.argmax(probs_row))
        conf = float(probs_row[top])
        info = {
            "predicted_class": self.class_names[top],
            "confidence": round(conf, 4),
            "confidence_threshold": self.conf_thr,
            "clip_p_poultry": None if clip_p is None else round(float(clip_p), 4),
            "clip_threshold": None if self.clip_thr is None else round(self.clip_thr, 4),
            "clip_best_match": clip_top,
            "feature_outlier_score": None if maha_s is None else round(float(maha_s), 4),
            "feature_outlier_threshold": None if self.maha_thr is None else round(self.maha_thr, 4),
        }
        if clip_p is not None and clip_p < self.clip_thr:
            return VERDICT_NOT_POULTRY, info
        if maha_s is not None and maha_s > self.maha_thr:
            return VERDICT_OOD, info
        if conf < self.conf_thr:
            return VERDICT_UNCERTAIN, info
        return VERDICT_PREDICTION, info

    def analyze(self, image, want_cam=True):
        pil = load_rgb(image)
        x = self._prepare([pil])
        with self.lock:
            f, p = self.multi(x, training=False)
        probs = np.asarray(p)[0]
        maha_s = float(self.maha.score(np.asarray(f))[0]) if self.maha is not None else None
        clip_p = clip_top = None
        if self.clip is not None:
            cp, ct = self.clip.score([pil])
            clip_p, clip_top = float(cp[0]), ct[0]
        verdict, info = self.decide(probs, maha_s, clip_p, clip_top)
        cam_img = None
        if want_cam and verdict in (VERDICT_PREDICTION, VERDICT_UNCERTAIN):
            try:
                with self.lock:
                    cam, _ = gradcam_heatmap(self.model, x)
                cam_img = overlay_cam(pil, cam)
            except Exception as e:  # noqa: BLE001
                self.warnings.append(f"Grad-CAM failed: {type(e).__name__}")
        return {"verdict": verdict, "info": info,
                "probs": {c: float(probs[i]) for i, c in enumerate(self.class_names)},
                "cam": cam_img}

    @staticmethod
    def pretty(name):
        return name.replace("_", " ")

    def message(self, res):
        v, info = res["verdict"], res["info"]
        if v == VERDICT_NOT_POULTRY:
            return ("### Not recognised as poultry droppings\n"
                    f"The image looks more like **{info['clip_best_match']}**. "
                    "Please upload a close-up, well-lit photo of chicken droppings.\n\n"
                    f"_{DISCLAIMER}_")
        if v == VERDICT_OOD:
            return ("### Image is unfamiliar to the model\n"
                    "It does not resemble the poultry-dropping photos the classifier was trained on "
                    "(possible other animal, different subject, or unusual lighting). "
                    "No diagnosis is given.\n\n" f"_{DISCLAIMER}_")
        name = self.pretty(info["predicted_class"])
        if v == VERDICT_UNCERTAIN:
            return (f"### Uncertain - leading guess: {name} ({info['confidence']*100:.1f}%)\n"
                    f"Confidence is below the {self.conf_thr*100:.0f}% reliability threshold. "
                    "Retake the photo or consult a veterinarian.\n\n" f"_{DISCLAIMER}_")
        return (f"### Screening result: **{name}** ({info['confidence']*100:.1f}% confidence)\n"
                f"_{DISCLAIMER}_")
