1) create .env with AWS credentials and secret in root folder

2) create data folder

    2.1) create notebook folder and save notebook there

3) exceute code to download dataset from S3

3) create folders:

    2.1) api / model / tests / upload / stress_test

3) 


Run docker


docker build -t preparation .            

sudo gpasswd -a $cremerf docker

docker run --rm --net host --gpus all -it \
    -v $(pwd):/home/src/app \
    -v /home/cremerf/FinalProject/data:/home/src/data \
    --workdir /home/src \
    preparation \
    bash