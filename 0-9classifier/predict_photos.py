from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage
import torch
import matplotlib.pyplot as plt
import train

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "photo_results"

def preprocess(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    im.thumbnail((756, 1008))
    a = np.asarray(im, dtype=np.float32)
    background = np.asarray(im.filter(ImageFilter.GaussianBlur(15)), dtype=np.float32)
    ink = np.maximum(background-a, 0)
    mask = ink > 35
    labels, count = ndimage.label(ndimage.binary_closing(mask, iterations=4))
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    component = labels == sizes.argmax()
    ys, xs = np.where(component)
    box = (max(0,xs.min()-4), max(0,ys.min()-4), min(a.shape[1],xs.max()+5), min(a.shape[0],ys.max()+5))
    x0,y0,x1,y1 = box
    crop = ink[y0:y1,x0:x1]
    crop = np.where(crop > 20, crop, 0)
    crop = np.clip(crop / max(float(np.percentile(crop[crop>0], 95)),1) * 255,0,255).astype(np.uint8)
    digit = Image.fromarray(crop)
    digit.thumbnail((20,20), Image.Resampling.LANCZOS)
    canvas = Image.new("L",(28,28)); canvas.paste(digit,((28-digit.width)//2,(28-digit.height)//2))
    arr = np.asarray(canvas,dtype=np.float32)/255
    cy,cx = ndimage.center_of_mass(arr)
    arr = ndimage.shift(arr,(13.5-cy,13.5-cx),order=1,mode="constant")
    return im, np.clip(arr,0,1)

def main():
    OUT.mkdir(exist_ok=True)
    checkpoint = torch.load(ROOT/"d2nn_results/best.pt", map_location="cpu", weights_only=True)
    train.NUM_LAYERS = checkpoint["model"]["phase"].shape[0]
    model = train.D2NN().to(train.DEVICE)
    model.load_state_dict(checkpoint["model"],strict=True); model.eval()
    files = sorted((ROOT/"test").glob("*.jpg"))
    processed = [preprocess(f) for f in files]
    inputs = torch.from_numpy(np.stack([a for _,a in processed]))[:,None].to(train.DEVICE)
    with torch.inference_mode():
        energies,_ = model(inputs)
        scores = train.loss_fn(energies).cpu().numpy()
    results=[]
    fig,axes=plt.subplots(len(files),3,figsize=(11,4*len(files)),squeeze=False)
    for i,(f,(original,a),score) in enumerate(zip(files,processed,scores)):
        Image.fromarray((a*255).astype(np.uint8)).save(OUT/(f.stem+"_input.png"))
        ranking=np.argsort(score)[::-1]
        result={"file":f.name,"prediction":int(ranking[0]),"detector_energy_shares":score.tolist()}
        results.append(result)
        axes[i,0].imshow(original,cmap="gray");axes[i,0].set_title(f.name);axes[i,0].axis("off")
        axes[i,1].imshow(a,cmap="gray",vmin=0,vmax=1);axes[i,1].set_title("Model input (28 x 28)");axes[i,1].axis("off")
        axes[i,2].bar(range(10),score);axes[i,2].set_xticks(range(10));axes[i,2].set_ylim(0,1);axes[i,2].set_title(f"Prediction: {ranking[0]} | energy share: {score[ranking[0]]:.1%}")
    fig.tight_layout();fig.savefig(OUT/"predictions.png",dpi=140);plt.close(fig)
    report={"checkpoint":"d2nn_results/best.pt","epoch":checkpoint.get("epoch"),"validation_accuracy":checkpoint.get("val_accuracy"),"results":results}
    (OUT/"predictions.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()

