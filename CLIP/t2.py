import clip
 
model, preprocess = clip.load("ViT-B/32", device="cuda")  # GPU加载
 
# model, preprocess = clip.load("ViT-B/32", device="cpu") # CPU加载
 
print("模型加载成功！")