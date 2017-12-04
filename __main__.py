from Preprocess import Preprocess

def main():
    data = Preprocess()

    data.load_data()

    data.preprocess_images()

    data.one_hot_encode_labels()

    train_x, train_y = data.get_training_data()

    print(train_y[0])

if __name__ == "__main__":
    main()
