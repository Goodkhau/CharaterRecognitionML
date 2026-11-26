"""
Draw a digit or letter and see the model's top-5 predictions.

Setup:
    pip install tensorflow pillow numpy

Before running, save your trained model:
    model = tf.keras.Model(inputs=input, outputs=outputs)
    model.save("model.keras")
"""
import tkinter as tk
from tkinter import ttk

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# ---------------- Settings ----------------
MODEL_PATH = "/home/gokhau/Desktop/CharaterRecognitionML/model_outputs/num_lett/best_model.keras"

# EMNIST "byclass" order: 0-9, A-Z, a-z (62 classes).
# Change this if your labels are ordered differently.
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# EMNIST images are stored transposed. If you trained on raw EMNIST
# (e.g. tensorflow_datasets) and did NOT fix the orientation, keep True.
# If your training images look upright when plotted, set to False.
TRANSPOSE = False

# Set to False if you did not divide pixels by 255 during training.
NORMALIZE_0_1 = True

CANVAS_SIZE = 280   # drawing area in pixels
BRUSH = 14          # pen thickness
TOP_K = 5
# ------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image):
    """Crop to the stroke, scale to 20px, center by mass in a 28x28 frame
    (same style as MNIST/EMNIST). Returns (1, 28, 28, 1) or None if blank."""
    bbox = img.getbbox()
    if bbox is None:
        return None

    digit = img.crop(bbox)
    w, h = digit.size
    scale = 20.0 / max(w, h)
    digit = digit.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
    )

    # Center of mass of the resized glyph
    arr = np.asarray(digit, dtype=np.float32)
    ys, xs = np.indices(arr.shape)
    total = arr.sum()
    cy = (ys * arr).sum() / total
    cx = (xs * arr).sum() / total

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, (int(round(14 - cx)), int(round(14 - cy))))

    x = np.asarray(canvas, dtype=np.float32)
    if NORMALIZE_0_1:
        x /= 255.0
    if TRANSPOSE:
        x = x.T
    return x.reshape(1, 28, 28, 1)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Draw a digit or letter")

        left = ttk.Frame(root, padding=10)
        left.grid(row=0, column=0)
        right = ttk.Frame(root, padding=10)
        right.grid(row=0, column=1, sticky="n")

        self.canvas = tk.Canvas(
            left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair", highlightthickness=1,
        )
        self.canvas.pack()
        ttk.Button(left, text="Clear", command=self.clear).pack(pady=(8, 0))

        ttk.Label(right, text="Top 5 predictions",
                  font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0, 8))

        self.rows = []
        for _ in range(TOP_K):
            row = ttk.Frame(right)
            row.pack(fill="x", pady=4)
            char = ttk.Label(row, text="–", width=2, font=("Helvetica", 22, "bold"))
            char.pack(side="left")
            bar = ttk.Progressbar(row, length=180, maximum=100)
            bar.pack(side="left", padx=8)
            pct = ttk.Label(row, text="", width=7)
            pct.pack(side="left")
            self.rows.append((char, bar, pct))

        # Off-screen PIL image mirrors what is drawn on the canvas
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.img)
        self.last = None

        self.canvas.bind("<Button-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

    def on_down(self, e):
        self.last = (e.x, e.y)
        self.stroke(e.x, e.y, e.x, e.y)

    def on_move(self, e):
        if self.last:
            self.stroke(*self.last, e.x, e.y)
            self.last = (e.x, e.y)

    def on_up(self, _):
        self.last = None
        self.predict()

    def stroke(self, x0, y0, x1, y1):
        r = BRUSH / 2
        self.canvas.create_line(x0, y0, x1, y1, fill="white",
                                width=BRUSH, capstyle="round", smooth=True)
        self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r,
                                fill="white", outline="white")
        self.draw.line([x0, y0, x1, y1], fill=255, width=BRUSH)
        self.draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=255)

    def predict(self):
        x = preprocess(self.img)
        if x is None:
            return
        probs = model.predict(x, verbose=0)[0]
        top = np.argsort(probs)[::-1][:TOP_K]
        for (char, bar, pct), idx in zip(self.rows, top):
            p = float(probs[idx]) * 100
            char.config(text=LABELS[idx])
            bar["value"] = p
            pct.config(text=f"{p:.1f}%")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        for char, bar, pct in self.rows:
            char.config(text="–")
            bar["value"] = 0
            pct.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
