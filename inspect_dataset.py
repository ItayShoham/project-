import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    with h5py.File('connect4_large_dataset.h5', 'r') as hf:
        features = np.array(hf['features'])
        labels = np.array(hf['labels']).flatten()
    return features, labels

def inspect_dataset():
    features, labels = load_dataset()
    print(f"Number of samples in the dataset: {features.shape[0]}")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")

def check_class_balance():
    _, labels = load_dataset()

    plt.hist(labels, bins=np.arange(8) - 0.5, edgecolor='black')
    plt.xticks(range(7))
    plt.xlabel('Move (Column)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Moves in the Dataset')
    plt.show()

def visualize_game_states():
    features, labels = load_dataset()

    for i in range(5):  # Visualize first 5 game states
        game_state = features[i]
        game_state_image = np.argmax(game_state, axis=0)  # Convert one-hot encoding to single channel

        plt.imshow(game_state_image, cmap='rainbow', interpolation='nearest')
        plt.title(f"Label (Move): {labels[i]}")
        plt.show()

def check_duplicates():
    features, _ = load_dataset()

    unique_states = np.unique(features, axis=0)
    print(f"Number of unique game states: {unique_states.shape[0]}")
    print(f"Total number of game states: {features.shape[0]}")

    if unique_states.shape[0] < features.shape[0]:
        print("The dataset contains duplicate game states.")
    else:
        print("The dataset does not contain duplicate game states.")

def verify_labels():
    features, labels = load_dataset()

    # Inspect a few game states and their corresponding labels
    for i in range(5):
        print(f"Game state {i}:")
        print(features[i])
        print(f"Label: {labels[i]}")

if __name__ == "__main__":
    inspect_dataset()
    check_class_balance()
    visualize_game_states()
    check_duplicates()
    verify_labels()
