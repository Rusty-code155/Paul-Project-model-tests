# Code by Turner Miles Peeples
# controller.py
from excel_handler import load_excel_files_from_folder, save_predictions_to_excel
from data_preprocessing import group_dataframes
from neural_network import ANNGCModel
from logger import log_feedback
from gui_feedback import run_feedback_gui

def run_training():
    print("Loading training data from 'practice' folder...")
    try:
        training_dfs = load_excel_files_from_folder('practice')
        grouped_data = group_dataframes(training_dfs)
        if not grouped_data:
            raise ValueError("No valid sheets found with temperature/conductivity data.")

        for group_id, df in grouped_data.items():
            print(f"\nTraining group: {group_id}")
            # Prepare features and target based on the class array
            X_df = df[["Shape", "Size (nm)", "Concentration", "Temperature"]]
            y = df["Thermal Conductivity"].values
            
            # Debug: Log the shape and columns of X_df
            print(f"Shape of X_df before preprocessing: {X_df.shape}")
            print(f"Columns in X_df: {X_df.columns.tolist()}")
            
            model = ANNGCModel(group_id)
            
            # Train the model
            model.train(X_df, y)
            
            # Make predictions
            preds = model.predict(X_df)
            error_percent = model.calculate_error(y, preds)
            plot_success = error_percent < 1.5
            
            max_attempts = 5
            attempt = 1
            while not plot_success and attempt <= max_attempts:
                print(f"Attempt {attempt}/{max_attempts} to improve model for group {group_id}")
                model.train(X_df, y)  # Retrain with same parameters
                preds = model.predict(X_df)
                error_percent = model.calculate_error(y, preds)
                plot_success = error_percent < 1.5
                attempt += 1
            
            if plot_success:
                run_feedback_gui(model, group_id)
                model.save_model()
                log_feedback(group_id, model.get_score(), model.history, phase="training")
                save_predictions_to_excel(df, preds, group_id, output_folder="output")
            else:
                print(f"Failed to achieve <1.5% error for group {group_id} after {max_attempts} attempts.")

    except Exception as e:
        print("[FATAL] Failed training due to:")
        print(e)
        raise

def run_testing():
    print("Loading testing data from 'testing' folder...")
    try:
        testing_dfs = load_excel_files_from_folder('testing')
        grouped_data = group_dataframes(testing_dfs)
        if not grouped_data:
            raise ValueError("No valid sheets found with temperature/conductivity data.")

        for group_id, df in grouped_data.items():
            print(f"\nTesting group: {group_id}")
            X_df = df[["Shape", "Size (nm)", "Concentration", "Temperature"]]
            y = df["Thermal Conductivity"].values
            
            # Debug: Log the shape and columns of X_df
            print(f"Shape of X_df before preprocessing: {X_df.shape}")
            print(f"Columns in X_df: {X_df.columns.tolist()}")
            
            model = ANNGCModel(group_id)
            preds = model.predict(X_df)
            error_percent = model.calculate_error(y, preds)
            plot_success = error_percent < 1.5
            if plot_success:
                run_feedback_gui(model, group_id)
                model.save_model()
                log_feedback(group_id, model.get_score(), model.history, phase="testing")
                save_predictions_to_excel(df, preds, group_id, output_folder="testing_output")

    except Exception as e:
        print("[FATAL] Failed testing due to:")
        print(e)
        raise