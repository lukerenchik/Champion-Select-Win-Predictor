import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ChampionUtilities.ChampionTranslator import ChampionTranslator
from data_processing.dataset import LoLMatchDataset
from data_processing.match_data_processor import MatchDataProcessor, normalize_name
from models.match_predictor import LoLMatchPredictor
import datetime



def main():
    # Constants
    CSV_FILE = '/home/lightbringer/Documents/Dev/Champion-Select-Win-Predictor/Champion-Select-Win-Predictor/ChampionSelectWinPredictor/data/lol_match_data.csv'
    CHAMPION_ATTRIBUTES_FILE = '/home/lightbringer/Documents/Dev/Champion-Select-Win-Predictor/Champion-Select-Win-Predictor/ChampionSelectWinPredictor/data/LoL-Champions.csv'
    BATCH_SIZE = 64 # 64
    EMBEDDING_DIM = 200 # 200
    NUM_EPOCHS = 8 # 10
    LEARNING_RATE = 0.000025 # 0.00001

    # Load and preprocess champion attributes
    champion_attributes_df = pd.read_csv(CHAMPION_ATTRIBUTES_FILE)
    champion_attributes_df['Id'] = champion_attributes_df['Id'].astype(str)
    champion_attributes = champion_attributes_df.set_index('Id').T.to_dict()

    # Initialize the translator
    translator = ChampionTranslator()

    # Initialize the data processor
    processor = MatchDataProcessor(CSV_FILE, champion_attributes, translator)

    # Access matches and unique champion IDs
    matches = processor.matches
    unique_champion_ids = processor.unique_champion_ids

    # Calculate the maximum champion ID for embedding layer size
    max_champion_id = max(unique_champion_ids)

    # Split into training, validation, and testing sets
    train_val_matches, test_matches = train_test_split(matches, test_size=0.1, random_state=42)
    train_matches, val_matches = train_test_split(train_val_matches, test_size=0.1, random_state=42)

    # Update the processor's matches for training set and convert matches to IDs with heuristics
    processor.matches = train_matches
    train_matches_data = processor.convert_matches_to_ids_with_heuristics()

    # Similarly for validation and test sets
    processor.matches = val_matches
    val_matches_data = processor.convert_matches_to_ids_with_heuristics()

    processor.matches = test_matches
    test_matches_data = processor.convert_matches_to_ids_with_heuristics()

    # Create datasets and loaders
    train_dataset = LoLMatchDataset(train_matches_data)
    val_dataset = LoLMatchDataset(val_matches_data)
    test_dataset = LoLMatchDataset(test_matches_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate heuristic dimension
    heuristic_dim = len(train_matches_data[0]['blue_heuristics'])

    # Initialize the model
    num_champions = max_champion_id  # Assuming champion IDs start at 1 and are sequential
    model = LoLMatchPredictor(
        num_champions=num_champions,
        embedding_dim=EMBEDDING_DIM,
        heuristic_dim=heuristic_dim
    )

    # Training setup
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
            blue_team_indices, red_team_indices, blue_heuristics, red_heuristics, labels = batch

            # Move data to the appropriate device
            blue_team_indices = blue_team_indices.to(device)
            red_team_indices = red_team_indices.to(device)
            blue_heuristics = blue_heuristics.to(device)
            red_heuristics = red_heuristics.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(blue_team_indices, red_team_indices, blue_heuristics, red_heuristics)

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
                blue_team_indices, red_team_indices, blue_heuristics, red_heuristics, labels = batch

                # Move data to the appropriate device
                blue_team_indices = blue_team_indices.to(device)
                red_team_indices = red_team_indices.to(device)
                blue_heuristics = blue_heuristics.to(device)
                red_heuristics = red_heuristics.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(blue_team_indices, red_team_indices, blue_heuristics, red_heuristics)

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
            blue_team_indices, red_team_indices, blue_heuristics, red_heuristics, labels = batch

            # Move data to the appropriate device
            blue_team_indices = blue_team_indices.to(device)
            red_team_indices = red_team_indices.to(device)
            blue_heuristics = blue_heuristics.to(device)
            red_heuristics = red_heuristics.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(blue_team_indices, red_team_indices, blue_heuristics, red_heuristics)

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

    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Format the test accuracy to two decimal places
    formatted_test_accuracy = f"{test_accuracy:.2f}"

    # Create the dynamic filename
    filename = f"win_predictor_{current_date}_{formatted_test_accuracy}.pth"

    # Save the trained model with the dynamic filename
    torch.save(model.state_dict(), filename)

if __name__ == '__main__':
    main()