# Code By Turner Miles Peeples
# neural_network.py
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class NanoFluidData:
    def __init__(self, ionic_fluid, nanoparticle, size, concentration, shape, temperature, thermal_conductivity):
        self.ionic_fluid = ionic_fluid
        self.nanoparticle = nanoparticle
        self.size = size
        self.concentration = concentration
        self.shape = str(shape)
        self.temperature = temperature
        self.thermal_conductivity = thermal_conductivity

    def to_array(self):
        return [self.ionic_fluid, self.nanoparticle, self.size, self.concentration, self.shape, self.temperature]

class ANNGCModel:
    def __init__(self, group_id):
        self.group_id = group_id
        safe_group_id = "_".join(str(g) for g in group_id)
        safe_group_id = re.sub(r'[<>:"/\\|?*\[\]()]+', '_', safe_group_id)
        self.model_path = f"models/model_{safe_group_id}.weights.h5"
        self.model = None
        self.input_dim = None  # Track the expected input dimension
        self.score = self.load_previous_score()
        self.history = []
        self.preprocessor = None
        self.scaler_y = StandardScaler()
        if os.path.exists(self.model_path) and self.score >= 0:
            self.load_model()

    def build_model(self, input_shape):
        print(f"Building new model with input_shape={input_shape}")
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),  # Add dropout to prevent overfitting
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def load_previous_score(self):
        log_path = "logs/training_feedback_log.csv"
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            group_str = str(self.group_id)
            group_data = df[df["Group ID"] == group_str]
            if not group_data.empty:
                return group_data["Score"].iloc[-1]
        return 0

    def preprocess_data(self, X_df, y=None, fit=True):
        numerical_cols = ["Size (nm)", "Concentration", "Temperature"]
        categorical_cols = ["Shape"]
        
        # Debug: Log the input DataFrame
        print(f"Input X_df columns: {X_df.columns.tolist()}")
        print(f"Input X_df shape: {X_df.shape}")
        
        # Fix SettingWithCopyWarning by using .loc
        X_df.loc[:, "Shape"] = X_df["Shape"].astype(str)
        
        if fit or self.preprocessor is None:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                    ('num', StandardScaler(), numerical_cols)
                ])
            if y is not None:
                X_transformed = self.preprocessor.fit_transform(X_df)
                y_transformed = self.scaler_y.fit_transform(y.reshape(-1, 1))
                
                # Debug: Log the shape and feature names
                print(f"Shape of preprocessed X: {X_transformed.shape}")
                print(f"Feature names after preprocessing: {self.preprocessor.get_feature_names_out()}")
                
                return X_transformed, y_transformed.flatten()
            else:
                X_transformed = self.preprocessor.fit_transform(X_df)
                
                # Debug: Log the shape and feature names
                print(f"Shape of preprocessed X: {X_transformed.shape}")
                print(f"Feature names after preprocessing: {self.preprocessor.get_feature_names_out()}")
                
                return X_transformed
        else:
            if y is not None:
                X_transformed = self.preprocessor.transform(X_df)
                y_transformed = self.scaler_y.transform(y.reshape(-1, 1))
                
                # Debug: Log the shape
                print(f"Shape of preprocessed X: {X_transformed.shape}")
                
                return X_transformed, y_transformed.flatten()
            else:
                X_transformed = self.preprocessor.transform(X_df)
                
                # Debug: Log the shape
                print(f"Shape of preprocessed X: {X_transformed.shape}")
                
                return X_transformed

    def train(self, X_df, y):
        # Preprocess the data
        X, y_transformed = self.preprocess_data(X_df, y, fit=True)
        
        # Determine the input dimension
        current_input_dim = X.shape[1]
        print(f"Current input dimension: {current_input_dim}")
        
        # If the model exists and the input dimension doesn't match, rebuild the model
        if self.model is not None and self.input_dim != current_input_dim:
            print(f"Input dimension mismatch! Expected {self.input_dim}, got {current_input_dim}. Rebuilding model.")
            self.model = self.build_model(current_input_dim)
            self.input_dim = current_input_dim
        elif self.model is None:
            # If no model exists, build a new one
            self.model = self.build_model(current_input_dim)
            self.input_dim = current_input_dim
        
        # Split into training and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
        
        # Get the indices of the training and validation sets to align Temperature values
        indices = np.arange(len(X_df))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Extract Temperature values in the original order
        temperatures = X_df["Temperature"].values
        temp_train = temperatures[train_indices]
        temp_val = temperatures[val_indices]
        
        # Train the model
        history = self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_val, y_val))
        self.history.append(history.history['loss'])
        
        # Make predictions on the training and validation sets
        y_train_pred = self.model.predict(X_train).flatten()
        y_val_pred = self.model.predict(X_val).flatten()
        
        # Inverse transform the predictions and true values
        y_train_true = self.scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_val_true = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        y_val_pred = self.scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
        
        # Combine training and validation data
        y_true_all = np.concatenate([y_train_true, y_val_true])
        y_pred_all = np.concatenate([y_train_pred, y_val_pred])
        temp_all = np.concatenate([temp_train, temp_val])
        
        # Plot all data in a single graph with Temperature as the x-axis
        self.plot_prediction_vs_actual(X, y_true_all, y_pred_all, temp_all, title="Prediction vs Actual (All Data)")

    def predict(self, X_df):
        X_transformed = self.preprocess_data(X_df, fit=False)
        preds_scaled = self.model.predict(X_transformed, verbose=0).flatten()
        return self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    def calculate_error(self, y_true, y_pred):
        errors = np.abs(y_true - y_pred) / y_true
        error_percent = np.mean(errors) * 100
        # Log actual vs predicted for debugging
        print(f"Actual vs Predicted for group {self.group_id}:")
        for actual, predicted in zip(y_true, y_pred):
            print(f"Actual: {actual:.4f}, Predicted: {predicted:.4f}, Error: {np.abs(actual - predicted) / actual * 100:.2f}%")
        return error_percent

    def plot_prediction_vs_actual(self, X, y_true, y_pred, temperatures, title="Prediction vs Actual"):
        """
        Plot predicted vs actual thermal conductivity values in a single graph.
        - Use Temperature as the x-axis.
        - Actual points in red, predicted points in blue.
        - Include lines of best fit for each set in matching colors.
        - Calculate and log error percentages.
        - Include group metadata (Ionic Fluid, Nanoparticle, Size, Concentration, Shape) in the title.
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        temperatures = np.array(temperatures)
    
        # Calculate percentage errors
        errors = np.abs((y_true - y_pred) / y_true) * 100
    
        # Log individual errors
        print(f"\nError Percentages for {title}:")
        for i, (true_val, pred_val, error) in enumerate(zip(y_true, y_pred, errors)):
            print(f"Sample {i}: True={true_val:.6f}, Predicted={pred_val:.6f}, Error={error:.2f}%")
    
        # Log average error
        avg_error = np.mean(errors)
        print(f"Average Error for {title}: {avg_error:.2f}%")
        if avg_error > 1.5:
            print("WARNING: Average error exceeds 1.5% threshold. Consider tuning the model or improving the data.")
    
        # Create a detailed title with group metadata
        ionic_fluid, nanoparticle, size, concentration, shape = self.group_id
        detailed_title = (
            f"{title}\n"
            f"Ionic Fluid: {ionic_fluid}, Nanoparticle: {nanoparticle}, "
            f"Size: {size} nm, Concentration: {concentration}, Shape: {shape}"
        )
    
        # Create a single plot
        plt.figure(figsize=(10, 8))
        
        # Plot actual points in red
        plt.scatter(temperatures, y_true, color='red', label='Actual', alpha=0.6)
        
        # Plot predicted points in blue
        plt.scatter(temperatures, y_pred, color='blue', label='Predicted', alpha=0.6)
        
        # Compute and plot line of best fit for actual points (red)
        z_actual = np.polyfit(temperatures, y_true, 1)  # Linear fit (degree 1)
        p_actual = np.poly1d(z_actual)
        plt.plot(temperatures, p_actual(temperatures), color='red', linestyle='--', label='Actual Best Fit')
        
        # Compute and plot line of best fit for predicted points (blue)
        z_pred = np.polyfit(temperatures, y_pred, 1)  # Linear fit (degree 1)
        p_pred = np.poly1d(z_pred)
        plt.plot(temperatures, p_pred(temperatures), color='blue', linestyle='--', label='Predicted Best Fit')
        
        # Customize the plot
        plt.xlabel('Temperature (K)')
        plt.ylabel('Thermal Conductivity (W/m*K)')
        plt.title(detailed_title, fontsize=10, pad=15)
        plt.legend()
        plt.grid(True)
        plt.show()
    
        return errors

    def update_score(self, feedback):
        if feedback == 'yes':
            self.score += 1
        elif feedback == 'no':
            self.score -= 1

    def get_score(self):
        return self.score

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        self.model.save_weights(self.model_path)

    def load_model(self):
        try:
            # Load the model weights without building the model first
            # We'll build the model in train() with the correct input shape
            if self.model is None:
                print("Model not built yet. Will build during training with correct input shape.")
            else:
                self.model.load_weights(self.model_path)
                self.input_dim = self.model.input_shape[1]  # Update input_dim after loading
                print(f"Model weights loaded from {self.model_path} with input_dim={self.input_dim}")
        except Exception as e:
            print(f"Failed to load model weights: {e}")
            print("Will build a new model during training.")
            self.model = None