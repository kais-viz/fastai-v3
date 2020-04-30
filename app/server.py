import aiohttp
import asyncio
import uvicorn
import requests
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?id=1mhsXSg9e6_pRfyDaig2PV_SciwWKlLf7&export=download'
export_file_name = 'export.pkl'

classes = ['black', 'blue', 'brown', 'dress', 'green', 'hoodie', 'pants', 'pink', 'red', 
		'shirt', 'shoes', 'shorts', 'silver', 'skirt', 'suit', 'white', 'yellow']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))

    _, _, pred_pct = learn.predict(img)
    prediction = get_preds(pred_pct, classes)
    return JSONResponse({'result': str(prediction)})

@app.route('/url_analyze', methods=['POST'])
async def url_analyze(url):
    # url = "https://live.staticflickr.com/8188/28638701352_1aa058d0c6_b.jpg" 
    response = await requests.get(url).content #get request contents

    img = await open_image(BytesIO(response)) #convert to image

    _, _, pred_pct = learn.predict(img) #predict while ignoring first 2 array inputs
    prediction = get_preds(pred_pct, classes)
    return JSONResponse({'result': str(prediction)})

def get_preds(obj, classes):
    predictions = {}
    x=0
    for item in obj:
        acc= round(item.item(), 3)*100
        if acc > 15:
            predictions[classes[x]] = acc
        x+=1
    predictions ={k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

    return predictions

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
