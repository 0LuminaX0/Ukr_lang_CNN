import os
import numpy as np
import tensorflow as tf
import pandas as pd
import tkinter as tk
from tkinter import Canvas, Frame, Button
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model
model = tf.keras.models.load_model("./data/ukrlang_recognition_mlp.keras")
#model = tf.keras.models.load_model("ukrlang_recognition_no_aug_mlp.keras")

letters = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 
           'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я', 'є', 'і', 'ї', 'ґ']


DRAW_SIZE = 280
MODEL_INPUT_SIZE = 28
BRUSH_RADIUS = 6

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognition")

        main_frame = Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas_frame = Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)

        canvas_border_frame = Frame(canvas_frame, width=DRAW_SIZE, height=DRAW_SIZE, bg='black', bd=2)
        canvas_border_frame.pack_propagate(False)
        canvas_border_frame.pack(padx=10, pady=10)

        self.canvas = Canvas(canvas_border_frame, width=DRAW_SIZE, height=DRAW_SIZE, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (DRAW_SIZE, DRAW_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.clear_button = Button(canvas_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(pady=5)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.process_and_predict)

        self.plot_frame = Frame(main_frame)
        self.plot_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.fig = None
        self.process_and_predict()

    def paint(self, event):
        x1, y1 = event.x - BRUSH_RADIUS, event.y - BRUSH_RADIUS
        x2, y2 = event.x + BRUSH_RADIUS, event.y + BRUSH_RADIUS
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (DRAW_SIZE, DRAW_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.process_and_predict()

    def process_and_predict(self, event=None):
        img_resized = self.image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BICUBIC)
        self.img_gray = img_resized.convert('L')
        img_inverted = ImageOps.invert(self.img_gray)

        img_array = np.array(img_inverted).astype('float32') / 255.0
        img_array = img_array.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1)
        predictions = model.predict(img_array)

        self.show_prediction(predictions)

    def show_prediction(self, predictions):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        letter_probabilities = predictions[0] * 100

        df = pd.DataFrame({'Letter': letters, 'Probability (%)': letter_probabilities})
        df = df.sort_values('Probability (%)', ascending=False)

        # Plot the predictions as a bar chart
        self.fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(df['Letter'], df['Probability (%)'], color='skyblue')
        plt.title("Prediction Probabilities")
        plt.xlabel("Letters")
        plt.ylabel("Probability (%)")
        plt.xticks(rotation=90)

        def on_click(event):
            for i, bar in enumerate(bars):
                if bar.contains(event)[0]:  # Check if bar was clicked
                    self.save_letter_image(df['Letter'].iloc[i])
                    break

        self.fig.canvas.mpl_connect('button_press_event', on_click)

        canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def save_letter_image(self, letter):
        os.makedirs("new_letters", exist_ok=True)

        existing_files = [f for f in os.listdir("new_letters") if f.startswith(letter)]
        next_number = len(existing_files) + 1
        filename = f"{letter}_{next_number:02d}.png"
        save_path = os.path.join("new_letters", filename)

        self.img_gray.save(save_path)
        print(f"Saved image as {save_path}")

# Main application
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()