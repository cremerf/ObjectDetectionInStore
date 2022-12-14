## Run docker

# Build image from folder

docker build -t preparation_fc .            

sudo gpasswd -a $cremerf docker

## Build container in AWS with image running

docker run --rm --net host --gpus all -it \
    -v /home/cremerf/FinalProject/training:/home/src/app \
    -v /home/eudesz/final_project/data:/home/src/dataset \
    -v /home/cremerf/FinalProject/data:/home/src/data \
    --workdir /home/src \
    preparation_fc \
    bash

## Build container in Local with image running

docker run --rm --net host --gpus all -it \
    -v $(pwd):/home/src/app \
    --workdir /home/src \
    preparation \
    bash

## Download files from server into local folder

scp -i ~/.ssh/id_rsa cremerf@ec2-3-135-65-134.us-east-2.compute.amazonaws.com:/home/cremerf/FinalProject/data .

scp -i ~/.ssh/id_rsa cremerf@ec2-3-135-65-134.us-east-2.compute.amazonaws.com:/home/cremerf/FinalProject/data/first_training .

## download data folder
scp -r cremerf@ec2-3-135-65-134.us-east-2.compute.amazonaws.com:/home/cremerf/FinalProject/data .
## download first training with weights folder
scp -r cremerf@ec2-3-135-65-134.us-east-2.compute.amazonaws.com:/home/cremerf/FinalProject/data/first_training .


## Train model in container

python3 train.py --img 416 --batch 4 --epochs 5 \
    --data /home/src/app/config_blmodel.yaml \
    --weights yolov5s.pt --cache --project /home/src/data \
    --name first_training


sudo chmod 777 -R predictions/