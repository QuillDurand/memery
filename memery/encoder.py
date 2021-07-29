__all__ = ['device', 'model', 'image_encoder', 'text_encoder', 'image_query_encoder']


import torch
import clip
from tqdm import tqdm
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = clip.load("ViT-B/16", device, jit=False)
model = model.float()


def image_encoder(img_loader, device):
    image_embeddings = torch.tensor(())
    print("Encoding images")
    # i = 0
    with torch.no_grad():
        for images, labels in tqdm(img_loader):
            try:
                batch_features = model.encode_image(images.to(device)).to('cpu')
                image_embeddings = torch.cat((image_embeddings, batch_features))
                # del batch_features
                # i += 1
                # if i%100 == 0:
                #     gc.collect()
                #     torch.cuda.empty_cache()
            except Exception as e:
                print("failed to encode:", e)
    gc.collect()
    torch.cuda.empty_cache()
    print("Done encoding images")
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    return(image_embeddings)


def text_encoder(text, device):
    with torch.no_grad():
        text = clip.tokenize(text).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return(text_features)


def image_query_encoder(image, device):
    with torch.no_grad():
        image_embed = model.encode_image(image.unsqueeze(0).to(device))
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    return(image_embed)