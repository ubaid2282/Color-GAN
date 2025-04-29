from flask import Flask, render_template, request, send_file
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Define your Generator model ===
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# === Load model ===
model = Generator()
model.load_state_dict(torch.load("color_gan_generator.pth", map_location="cpu"))
model.eval()

# === Image transform ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg")
            file.save(img_path)

            # Open and process image
            img = Image.open(img_path).convert("L")
            img_tensor = transform(img).unsqueeze(0)  # shape: (1, 1, 32, 32)

            with torch.no_grad():
                output_tensor = model(img_tensor)
                output_tensor = (output_tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1)  # de-normalize
                output_img = transforms.ToPILImage()(output_tensor)
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
                output_img.save(output_path)

            return render_template("index.html", output_image="output.jpg")

    return render_template("index.html", output_image=None)

if __name__ == "__main__":
    app.run(debug=True)
