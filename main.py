# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.utils import get_color_from_hex
from kivy.metrics import dp
from kivy.logger import Logger
from kivy.graphics.texture import Texture
from kivy.clock import Clock

# Import other dependencies
import cv2
import tensorflow as tf
from PIL import Image as PILImage
import numpy as np
import os
import time

class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, 0.6), pos_hint={'center_y': 0.5})  # Centered, taking 60% height
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, 0.4), )
        self.button1 = Button(text="Verify in video", on_press=self.toggle_video_verification, size_hint=(1, 0.4), )
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, 0.2))  # Taking 20% height
        self.url = "http://192.168.233.205:8080/video" 

        # Set background color of button1 to green
        self.button1.background_color = get_color_from_hex('#00FF00')  # Green color

        # Add buttons to a horizontal box layout with padding and spacing
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.4), )  # Taking 20% height
        button_layout.padding = dp(40)  # Left and right padding
        button_layout.spacing = dp(40)  # Gap between buttons
        button_layout.add_widget(self.button)
        button_layout.add_widget(self.button1)

        # Add items to main vertical layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(button_layout)
        layout.add_widget(self.verification_label)

        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path='quantized_model.tflite')
        self.interpreter.allocate_tensors()

        # Setup video capture device
        self.capture = cv2.VideoCapture(self.url)  # Assuming webcam index is 0
        self.video_verification_active = False  # Flag to indicate if video verification is active
        Clock.schedule_interval(self.update, 1.0 / 12.0)
        # Counter for iterations
        self.video_verification_iterations = 0

        return layout

    def update(self, *args):
        # Read frame from OpenCV
        ret, frame = self.capture.read()

        # Flip horizontally and convert image to texture
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture
        else:
            Logger.warning("Failed to capture frame from the camera.")


    def preprocess_and_predict(self, image_path):
        # Read the image using PIL
        img = PILImage.open(image_path)
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0

        # Add a batch dimension
        input_data = np.expand_dims(img, axis=0)

        # Set the input tensor
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get the model output
        output_data = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])

        return output_data

    def verify(self, *args):
        import time

        start = time.time()
        # Capture input image from the webcam
        save_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(save_path, frame)

        # Perform inference using TensorFlow Lite model
        output_data = self.preprocess_and_predict(save_path)[0][0]
        end = time.time()
        #print(output_data)

        # Set verification text
        color = 'Green' if output_data < 0.5 else 'Red'
        self.verification_label.text = f"{color} {output_data*100:.2f}%"

        # Set text color based on the 'color' variable
        if color == 'Green':
            self.verification_label.color = (0, 1, 0, 1)  # Green color in RGBA format
        else:
            self.verification_label.color = (1, 0, 0, 1)  # Red color in RGBA format

        timer = end - start

        # Log out details
        Logger.info(f"Output values: {output_data}")
        Logger.info(f"Detection: {color}")
        Logger.info(f"Execution time: {timer*1000:.2f} ms")
        print("\n")

        return output_data, color, timer

    def toggle_video_verification(self, *args):
        if not self.video_verification_active:
            self.button1.text = "Stop Verification"
            self.button1.bind(on_press=self.toggle_video_verification)
            # Schedule the first iteration of verify_in_video
            Clock.schedule_once(self.verify_in_video, 0)
            self.video_verification_active = True
            # Set background color of button1 to Red
            self.button1.background_color = get_color_from_hex('#FF0000')  # Green color
        else:
            # Stop the clock if video_verification_active is False
            Clock.unschedule(self.verify_in_video)
            self.button1.text = "Verify in video"
            self.button1.bind(on_press=self.toggle_video_verification)
            self.video_verification_active = False
            self.video_verification_iterations = 0
            # Set background color of button1 to green
            self.button1.background_color = get_color_from_hex('#00FF00')  # Green color
            print("You pressed the stop button")


    def verify_in_video(self, dt):
        if self.video_verification_active:
            print(f"Iteration: {self.video_verification_iterations}")
            output_data, color, timer = self.verify()
            self.verification_label.text = f"{color} {output_data*100:.2f}%"
            # Schedule the next iteration of verify_in_video
            Clock.schedule_once(self.verify_in_video,1)
            self.video_verification_iterations += 1
            #time.sleep(1)


if __name__ == '__main__':
    CamApp().run()
