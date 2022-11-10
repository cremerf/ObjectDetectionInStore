1) create .env with AWS credentials and secret in root folder

2) create data folder

    2.1) create notebook folder and save notebook there

3) exceute code to download dataset from S3

3) create folders:

    2.1) api / model / tests / upload / stress_test

3) 


Run docker

docker build -t prepare_train .            

sudo gpasswd -a $eudesz docker

sudo docker run --rm --net host -it\
    -v $(pwd):/home/src/prepare_train \
    --workdir /home/src/prepare_train \
    402c17b27cca \
    bash