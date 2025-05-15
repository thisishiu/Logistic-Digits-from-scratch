import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
from process_image import *

class Model:
    def __init__(self):
        self.model = np.load("model.npz")

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, x):
        x = np.hstack((np.ones((1, 1)), x))  # add bias term
        scores = {c: self.sigmoid(x @ self.model[c])[0, 0] for c in self.model.files}
        total = sum(np.exp(v) for v in scores.values())
        softmax_scores = {c: np.exp(v) / total for c, v in scores.items()}
        return softmax_scores

    def predict(self, x):
        probs = self.softmax(x.reshape(1, -1))
        return max(probs, key=probs.get), round(probs[max(probs, key=probs.get)] * 100, 2)

A = Model()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw here")

        self.canvas_size = 280 
        self.image_size = 28

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.ok_button = tk.Button(self.button_frame, text="OK", command=self.predict)
        self.ok_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.clear)
        self.reset_button.pack(side=tk.LEFT)

        self.result_label = tk.Label(self.root, text="Draw and press OK", font=('Arial', 14))
        self.result_label.pack()

        # Label to show the resized image
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.result_label.config(text="Draw and press OK")

    def predict(self):
        # Resize to 28x28
        small_img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        small_img = ImageOps.invert(small_img) 
        img_array = np.array(small_img) / 255.0 
        img_array = re_center_image(img_array, size=28, xp=np)
        img_array = img_array.reshape(1, 28, 28)

        # Show the resized image in the Tkinter window
        img_display = Image.fromarray((img_array[0] * 255).astype(np.uint8))
        img_display = img_display.resize((280, 280), Image.Resampling.NEAREST)
        tk_img = ImageTk.PhotoImage(img_display)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img  # Keep a reference
    
        # Call the model to predict
        img_array = img_array.reshape(1, -1)
        result, prob = A.predict(img_array)
        self.result_label.config(text=f"{result} ({prob}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
