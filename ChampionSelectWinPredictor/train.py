# train.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ChampionSelectWinPredictor import (
    MatchDataProcessor,
    LoLMatchDataset,
    LoLMatchPredictor,
    normalize_name,
)

from ChampionUtilities.ChampionTranslator import ChampionTranslator

# Constants
CSV_FILE = 'data/lol_match_data.csv'
CHAMPION_ATTRIBUTES_FILE = 'data/LoL-Champions.csv'
BATCH_SIZE = 32
EMBEDDING_DIM = 100
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
CLASS_EMBEDDING_DIM = 12  # Choose appropriate value
DAMAGE_TYPE_DIM = 3

def main():
    # Load and preprocess champion attributes
    champion_attributes_df = pd.read_csv(CHAMPION_ATTRIBUTES_FILE)

    # Normalize 'Style' (1-10) to [0, 1]
    champion_attributes_df['Style'] = (champion_attributes_df['Style'] - 1) / (10 - 1)

    # Normalize other numerical features
    numerical_features = ['Difficulty', 'Damage', 'Sturdiness', 'Crowd-Control', 'Mobility', 'Functionality']
    for feature in numerical_features:
        champion_attributes_df[feature] = (champion_attributes_df[feature] - 1) / (3 - 1)

    # Encode 'Class' with IDs
    unique_classes = champion_attributes_df['Class'].unique()
    class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
    num_classes = len(unique_classes)
    champion_attributes_df['ClassID'] = champion_attributes_df['Class'].map(class_to_id)

    # One-Hot Encode 'DamageType'
    damage_type_to_one_hot = {
        'P': [1.0, 0.0, 0.0],
        'M': [0.0, 1.0, 0.0],
        'PM': [0.0, 0.0, 1.0]
    }
    champion_attributes_df['DamageTypeOneHot'] = champion_attributes_df['DamageType'].map(damage_type_to_one_hot)

    # Set 'Id' as string type to match champion IDs used in your mappings
    champion_attributes_df['Id'] = champion_attributes_df['Id'].astype(str)

    # Create a dictionary with 'Id' as keys
    champion_attributes = champion_attributes_df.set_index('Id').T.to_dict()

    # Initialize the translator
    translator = ChampionTranslator()

    # Initialize the data processor
    processor = MatchDataProcessor(CSV_FILE, champion_attributes, translator)

    # Access matches and unique champions
    matches = processor.matches
    unique_champions = processor.unique_champions

    # Create champion ID mappings
    champion_to_id = {normalize_name(champion): idx for idx, champion in enumerate(unique_champions)}
    num_champions = len(unique_champions)

    # Split into training, validation, and testing sets
    train_val_matches, test_matches = train_test_split(matches, test_size=0.1, random_state=42)
    train_matches, val_matches = train_test_split(train_val_matches, test_size=0.1, random_state=42)

    # Update the processor's matches for training set and convert matches to IDs
    processor.matches = train_matches
    train_matches_ids = processor.convert_matches_to_ids(champion_to_id)

    # Update the processor's matches for validation set and convert matches to IDs
    processor.matches = val_matches
    val_matches_ids = processor.convert_matches_to_ids(champion_to_id)

    # Update the processor's matches for test set and convert matches to IDs
    processor.matches = test_matches
    test_matches_ids = processor.convert_matches_to_ids(champion_to_id)

    # Create datasets and loaders
    train_dataset = LoLMatchDataset(train_matches_ids)
    val_dataset = LoLMatchDataset(val_matches_ids)
    test_dataset = LoLMatchDataset(test_matches_ids)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate attribute dimensions
    attribute_dim = 7  # Number of numerical features
    input_dim = (EMBEDDING_DIM + attribute_dim + CLASS_EMBEDDING_DIM + DAMAGE_TYPE_DIM) * 2

    # Initialize the model
    model = LoLMatchPredictor(
        num_champions=num_champions,
        embedding_dim=EMBEDDING_DIM,
        attribute_dim=attribute_dim,
        num_classes=num_classes,
        class_embedding_dim=CLASS_EMBEDDING_DIM,
        damage_type_dim=DAMAGE_TYPE_DIM
    )

    # Training code...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Unpack the batch
            (blue_team_ids, red_team_ids,
             blue_numerical, red_numerical,
             blue_class_ids, red_class_ids,
             blue_damage_one_hot, red_damage_one_hot,
             labels) = batch

            # Move data to the appropriate device
            blue_team_ids = blue_team_ids.to(device)
            red_team_ids = red_team_ids.to(device)
            blue_numerical = blue_numerical.to(device)
            red_numerical = red_numerical.to(device)
            blue_class_ids = blue_class_ids.to(device)
            red_class_ids = red_class_ids.to(device)
            blue_damage_one_hot = blue_damage_one_hot.to(device)
            red_damage_one_hot = red_damage_one_hot.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                blue_team_ids, red_team_ids,
                blue_numerical, red_numerical,
                blue_class_ids, red_class_ids,
                blue_damage_one_hot, red_damage_one_hot
            )

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * labels.size(0)

        # Calculate average loss over the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {epoch_loss:.4f}')

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                # Unpack the batch
                (blue_team_ids, red_team_ids,
                 blue_numerical, red_numerical,
                 blue_class_ids, red_class_ids,
                 blue_damage_one_hot, red_damage_one_hot,
                 labels) = batch

                # Move data to the appropriate device
                blue_team_ids = blue_team_ids.to(device)
                red_team_ids = red_team_ids.to(device)
                blue_numerical = blue_numerical.to(device)
                red_numerical = red_numerical.to(device)
                blue_class_ids = blue_class_ids.to(device)
                red_class_ids = red_class_ids.to(device)
                blue_damage_one_hot = blue_damage_one_hot.to(device)
                red_damage_one_hot = red_damage_one_hot.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(
                    blue_team_ids, red_team_ids,
                    blue_numerical, red_numerical,
                    blue_class_ids, red_class_ids,
                    blue_damage_one_hot, red_damage_one_hot
                )

                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

                # Apply threshold to get binary predictions
                predicted = (outputs >= 0.5).float()

                # Update counts
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n')

    print("Training complete.")

    # Evaluation on Test Set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            # Unpack the batch
            (blue_team_ids, red_team_ids,
             blue_numerical, red_numerical,
             blue_class_ids, red_class_ids,
             blue_damage_one_hot, red_damage_one_hot,
             labels) = batch

            # Move data to the appropriate device
            blue_team_ids = blue_team_ids.to(device)
            red_team_ids = red_team_ids.to(device)
            blue_numerical = blue_numerical.to(device)
            red_numerical = red_numerical.to(device)
            blue_class_ids = blue_class_ids.to(device)
            red_class_ids = red_class_ids.to(device)
            blue_damage_one_hot = blue_damage_one_hot.to(device)
            red_damage_one_hot = red_damage_one_hot.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                blue_team_ids, red_team_ids,
                blue_numerical, red_numerical,
                blue_class_ids, red_class_ids,
                blue_damage_one_hot, red_damage_one_hot
            )

            # Compute loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)

            # Apply threshold to get binary predictions
            predicted = (outputs >= 0.5).float()

            # Update counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for additional metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    # Calculate average test loss and accuracy
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100 * correct / total
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Optionally, calculate additional metrics like ROC AUC, Precision, Recall, F1-Score
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(all_labels, all_predictions)
    # Calculate Precision, Recall, F1-Score with threshold of 0.5
    binary_predictions = (all_predictions >= 0.5).astype(int)
    precision = precision_score(all_labels, binary_predictions)
    recall = recall_score(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)

    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()
