import torch
import clip
from PIL import Image
from torchvision import transforms



class ClipEmbeding:
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    def __init__(self):
        self.model, self.processor = clip.load("models/RN50.pt", device=self.device)
        self.tokenizer = clip.tokenize
 
    def probs(self, image: Image):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(["a diagram", "a dog", "a cat"]).to(self.device)
 
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(process_image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
        print("Label probs:", probs)
 
 
    def embeding(self, image: Image, text: str):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer([text]).to(self.device)
 
        image_features = self.model.encode_image(process_image)
        text_features = self.model.encode_text(text)
        return image_features, text_features
 
 
if __name__ == "__main__":
    image_path = 'CLIP/cat.png'
 
    pil_image = Image.open(image_path)
 
    clip_embeding = ClipEmbeding()
    clip_embeding.probs(pil_image)