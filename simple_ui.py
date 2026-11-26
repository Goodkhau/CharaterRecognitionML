"""
Draw a digit, get a prediction.
 
Setup:
    pip install tensorflow pillow numpy
 
Before running, save your trained model once:
    model.save("digit_model.keras")
 
Then:
    python digit_app.py [path/to/model.keras]
"""
import sys
import tkinter as tk
from tkinter import ttk, messagebox
 
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
 
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "./model_outputs/best_model.keras"
CANVAS_SIZE = 280   # on-screen drawing area (10x the model input)
BRUSH_RADIUS = 11   # pen thickness in canvas pixels
 
 
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Turn the 280x280 drawing (white on black) into a (1, 28, 28, 1) array.
 
    Mimics how MNIST was built: crop to the digit, fit it in a 20x20 box,
    then place it in a 28x28 frame centred by its centre of mass. This makes
    predictions much more reliable for hand-drawn input.
    """
    bbox = pil_img.getbbox()
    if bbox is None:  # nothing drawn
        return None
 
    digit = pil_img.crop(bbox)
    w, h = digit.size
    scale = 20.0 / max(w, h)
    digit = digit.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
    )
 
    frame = Image.new("L", (28, 28), 0)
    frame.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
 
    arr = np.asarray(frame, dtype="float32") / 255.0
 
    # Shift so the centre of mass sits at the middle of the image
    total = arr.sum()
    if total > 0:
        ys, xs = np.indices(arr.shape)
        cy, cx = (ys * arr).sum() / total, (xs * arr).sum() / total
        arr = np.roll(arr, (round(13.5 - cy), round(13.5 - cx)), axis=(0, 1))
 
    return arr.reshape(1, 28, 28, 1)
 
 
class DigitApp:
    def __init__(self, root: tk.Tk, model: tf.keras.Model):
        self.root = root
        self.model = model
        root.title("Digit Recognizer")
        root.resizable(False, False)
 
        # Off-screen image mirrors what's drawn on the canvas
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last = None
 
        main = ttk.Frame(root, padding=12)
        main.grid()
 
        # Left: canvas + buttons
        left = ttk.Frame(main)
        left.grid(row=0, column=0, padx=(0, 16))
        self.canvas = tk.Canvas(
            left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair", highlightthickness=1,
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        ttk.Button(left, text="Clear", command=self.clear).pack(fill="x", pady=(8, 0))
 
        # Right: prediction + per-class probabilities
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="n")
        ttk.Label(right, text="Prediction").pack()
        self.pred_label = ttk.Label(right, text="–", font=("Helvetica", 56, "bold"))
        self.pred_label.pack()
        self.conf_label = ttk.Label(right, text="Draw a digit 0–9")
        self.conf_label.pack(pady=(0, 10))
 
        self.bars = []
        for d in range(10):
            row = ttk.Frame(right)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=str(d), width=2).pack(side="left")
            bar = ttk.Progressbar(row, length=140, maximum=1.0)
            bar.pack(side="left", padx=4)
            pct = ttk.Label(row, text="0%", width=5)
            pct.pack(side="left")
            self.bars.append((bar, pct))
 
    # --- drawing -----------------------------------------------------------
    def on_press(self, e):
        self.last = (e.x, e.y)
        self.paint(e.x, e.y, e.x, e.y)
 
    def on_move(self, e):
        x0, y0 = self.last
        self.paint(x0, y0, e.x, e.y)
        self.last = (e.x, e.y)
 
    def on_release(self, _):
        self.last = None
        self.predict()
 
    def paint(self, x0, y0, x1, y1):
        r = BRUSH_RADIUS
        self.canvas.create_line(
            x0, y0, x1, y1, fill="white", width=r * 2,
            capstyle=tk.ROUND, smooth=True,
        )
        self.draw.line([x0, y0, x1, y1], fill=255, width=r * 2)
        self.draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=255)
 
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.pred_label.config(text="–")
        self.conf_label.config(text="Draw a digit 0–9")
        for bar, pct in self.bars:
            bar["value"] = 0
            pct.config(text="0%")
 
    # --- inference ---------------------------------------------------------
    def predict(self):
        x = preprocess(self.image)
        if x is None:
            return
        probs = self.model.predict(x, verbose=0)[0]
        best = int(np.argmax(probs))
        self.pred_label.config(text=str(best))
        self.conf_label.config(text=f"Confidence: {probs[best] * 100:.1f}%")
        for (bar, pct), p in zip(self.bars, probs):
            bar["value"] = float(p)
            pct.config(text=f"{p * 100:.0f}%")
 
 
def main():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Could not load model",
            f"Failed to load '{MODEL_PATH}'.\n\n{exc}\n\n"
            "Save your model first with model.save('digit_model.keras').",
        )
        return
 
    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()
 
 
if __name__ == "__main__":
    main()
 