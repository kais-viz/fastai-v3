# Deploying [fast.ai](https://www.fast.ai) models on AWS EC2


This repo can be used as a starting point to deploy [fast.ai](https://github.com/fastai/fastai) models on AWS EC2 instances.

My apparel app described here is up at `https://fastai-v3.onrender.com`. Test it out with clothing images!


## Setting up the app and model files

To set up the application, you will have to follow the fastai's [deploying using render](https://course.fast.ai/deployment_render.html#fork-the-starter-app-on-github) tutorial. You will have to [fork the starter app on github](https://course.fast.ai/deployment_render.html#fork-the-starter-app-on-github) and [follow the per-project setup](https://course.fast.ai/deployment_render.html#per-project-setup).

## Local testing

To run the app server locally, run this command in your terminal:

```
python app/server.py serve
```

If you have Docker installed, you can test your app in the same environment as Render’s by running the following command at the root of your repository:

```
docker build -t fastai-v3 . && docker run --rm -it -p 5000:5000 fastai-v3
```

Go to [http://localhost:5000/](http://localhost:5000/) to test your app.

## Set up AWS EC2 Instance


## Deploying the Model on AWS

To deploy the trained model, I used lankinen's approach and followed his tutorial in his [medium article](https://medium.com/@lankinen/fastai-model-to-production-this-is-how-you-make-web-app-that-use-your-model-57d8999450cf).

### Notes on the article
* When trying to install `torch_nightly`, the URL provided gives a 404 error, to get around it, visit the URL and manually find the latest version applicable to your environment (Ubunutu 18) and install it.

```
wget https://download.pytorch.org/whl/nightly/cpu/torch_nightly-1.2.0.dev20190805%2Bcpu-cp36-cp36m-linux_x86_64.whl
pip3 install torch_nightly-1.2.0.dev20190805%2Bcpu-cp36-cp36m-linux_x86_64.whl
```

* You might need to use t2.large instance which provides 8GB of RAM, and increasing the volume to 15GB because you might ran out of memory and disk space while installing fastai library.

* Just before loading up the server, you should update torchvision's installation by type:
`pip3 install torchvision`