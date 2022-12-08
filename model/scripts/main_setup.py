from packages.prepare_dataset import main_prepare_datasets
from packages.prepare_labels import main_prepare_labels


def main():
    main_prepare_datasets()
    main_prepare_labels()



if __name__ == '__main__':
    main()