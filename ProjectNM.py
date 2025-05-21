import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from google.colab.output import eval_js
from IPython.display import display, HTML
import io
import base64
from PIL import Image

# 1. Load and prepare the MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Add channel dimension (required for Conv2D layers)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# 2. Build a simple model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="adam", 
              metrics=["accuracy"])

# 3. Train the model
print("\nTraining model...")
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)

# 4. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# 5. Create drawing canvas
drawing_html = HTML('''
<div style="text-align: center;">
  <h2>Draw a digit (0-9)</h2>
  <canvas id="canvas" width="280" height="280" style="border: 2px solid; border-radius: 5px; margin: 0 auto; cursor: crosshair;"></canvas>
  <div style="margin: 10px 0;">
    <button id="clear-button" style="background-color: #ff6b6b; color: white; padding: 5px 10px; border: none; border-radius: 5px; cursor: pointer;">Clear</button>
    <button id="predict-button" style="background-color: #4ecdc4; color: white; padding: 5px 10px; border: none; border-radius: 5px; cursor: pointer; margin-left: 10px;">Predict</button>
  </div>
  <div id="result" style="margin-top: 20px; font-size: 24px; font-weight: bold;">Draw a digit and click Predict</div>
</div>

<script>
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const clearButton = document.getElementById('clear-button');
  const predictButton = document.getElementById('predict-button');
  const resultDiv = document.getElementById('result');
  
  // Set canvas background to white
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Setup drawing parameters
  ctx.lineWidth = 15;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'black';
  
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;
  
  // Drawing event listeners
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseout', stopDrawing);
  
  // Touch support for mobile devices
  canvas.addEventListener('touchstart', handleTouch);
  canvas.addEventListener('touchmove', handleTouch);
  canvas.addEventListener('touchend', stopDrawing);
  
  // Button handlers
  clearButton.addEventListener('click', clearCanvas);
  predictButton.addEventListener('click', predict);
  
  function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
  }
  
  function draw(e) {
    if (!isDrawing) return;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    [lastX, lastY] = [e.offsetX, e.offsetY];
  }
  
  function stopDrawing() {
    isDrawing = false;
  }
  
  function handleTouch(e) {
    e.preventDefault();
    if (e.type === 'touchstart') {
      isDrawing = true;
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      [lastX, lastY] = [touch.clientX - rect.left, touch.clientY - rect.top];
    } else if (e.type === 'touchmove' && isDrawing) {
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const y = touch.clientY - rect.top;
      
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(x, y);
      ctx.stroke();
      [lastX, lastY] = [x, y];
    }
  }
  
  function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultDiv.textContent = 'Draw a digit and click Predict';
  }
  
  function predict() {
    resultDiv.textContent = 'Processing...';
    const imageData = canvas.toDataURL('image/png');
    google.colab.kernel.invokeFunction('notebook.predict', [imageData], {});
  }
</script>
''')

# 6. Function to process drawn image and make prediction
def predict_digit(image_data):
    try:
        # Get the base64 encoded image data
        image_data = image_data.split(',')[1]
        
        # Convert to image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 (MNIST format)
        image = image.resize((28, 28))
        
        # Prepare for model input
        image_array = np.array(image)
        image_array = 255 - image_array  # Invert (MNIST has white digits on black)
        image_array = image_array / 255.0  # Normalize
        
        # Display the processed image
        plt.figure(figsize=(2, 2))
        plt.imshow(image_array, cmap='gray')
        plt.title('Processed Input')
        plt.axis('off')
        plt.show()
        
        # Reshape for prediction (add batch and channel dimensions)
        image_array = np.expand_dims(image_array, axis=(0, -1))
        
        # Make prediction
        predictions = model.predict(image_array)
        digit = np.argmax(predictions[0])
        confidence = float(predictions[0][digit])
        
        # Create bar chart of probabilities
        plt.figure(figsize=(8, 3))
        plt.bar(range(10), predictions[0])
        plt.xticks(range(10))
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title('Prediction Probabilities')
        plt.show()
        
        return f"Predicted: {digit} (Confidence: {confidence:.2%})"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Register the callback
from google.colab import output
output.register_callback('notebook.predict', predict_digit)

# Display drawing interface
display(drawing_html)

# Instructions
print("\n---- How to use ----")
print("1. Draw a digit (0-9) on the canvas")
print("2. Click 'Predict' to see what digit the model recognizes")
print("3. Click 'Clear' to try again")
print("-------------------")
