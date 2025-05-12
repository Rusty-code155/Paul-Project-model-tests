Thermal Conductivity Predictor
Written By: Turner Miles Peeples
=============================

This project predicts the thermal conductivity of nanofluids using experimental data from Excel files. It employs a neural network model for training and testing predictions, with graphical user interfaces (GUIs) for user interaction and feedback. The project processes data, trains models, generates predictions, and visualizes results.

Project Overview
---------------
The Thermal Conductivity Predictor processes Excel files containing nanofluid properties (e.g., Ionic Fluid, Nanoparticle, Size, Shape, Concentration, Temperature, and Thermal Conductivity). It groups data by unique combinations of these properties, trains an artificial neural network (ANN) for each group, and predicts thermal conductivity values. The project includes GUIs for training, testing, prediction, and visualizing model performance.

Libraries Used
-------------
The project relies on the following Python libraries:

- pandas: Reads, manipulates, and saves Excel files. Handles dataframes for data processing and grouping.
  Version: >=1.5.3
  Purpose: Data loading (pd.read_excel), manipulation, and export (to_excel).

- numpy: Provides numerical operations for arrays, used in calculations like error percentages and data preprocessing.
  Version: >=1.24.3
  Purpose: Array operations, mathematical computations.

- tensorflow: Powers the ANN model for training and prediction of thermal conductivity.
  Version: >=2.12.0
  Purpose: Building and training neural networks (keras.Sequential, layers.Dense).

- scikit-learn: Used for data preprocessing, including scaling numerical features (StandardScaler), encoding categorical variables (OneHotEncoder), and splitting data (train_test_split).
  Version: >=1.2.2
  Purpose: Preprocessing and data splitting.

- matplotlib: Generates plots for visualizing actual vs. predicted thermal conductivity values.
  Version: >=3.7.1
  Purpose: Plotting (plt.scatter, plt.plot) and embedding plots in GUIs.

- tkinter: Python's standard GUI library, used for creating interactive interfaces for training, testing, prediction, and dashboard visualization.
  Version: Included with Python
  Purpose: GUI creation (tk.Tk, ttk.Button, ttk.Label).

- difflib: Used for fuzzy matching of column names in Excel files to handle variations in header names.
  Version: Included with Python
  Purpose: Column name matching (difflib.get_close_matches).

- os: Handles file system operations like creating directories and listing files.
  Version: Included with Python
  Purpose: File and directory management (os.listdir, os.makedirs).

- re: Provides regular expressions for sanitizing file names.
  Version: Included with Python
  Purpose: String manipulation (re.sub).

- datetime: Used for timestamping logs.
  Version: Included with Python
  Purpose: Logging timestamps (datetime.now).

- csv: Handles writing feedback logs to CSV files.
  Version: Included with Python
  Purpose: Logging (csv.writer).

- ast: Safely evaluates string representations of Python literals in the dashboard for plotting loss history.
  Version: Included with Python
  Purpose: Parsing log data (ast.literal_eval).

Installation
------------
1. Install Python: Ensure Python 3.8 or higher is installed. Download from python.org.
2. Set Up a Virtual Environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies: Install required libraries using pip:
   pip install pandas numpy tensorflow scikit-learn matplotlib
   Note: tkinter, difflib, os, re, datetime, csv, and ast are included with Python.
4. Verify Installation: Ensure all libraries are installed by running:
   python -c "import pandas, numpy, tensorflow, sklearn, matplotlib, tkinter, difflib, os, re, datetime, csv, ast"

Project Structure
-----------------
- main.py: Entry point, launches a GUI to choose between training and testing modes.
- controller.py: Manages training and testing workflows, coordinating data loading, model training, and prediction saving.
- neural_network.py: Defines the ANN model (ANNGCModel) for training and prediction, including data preprocessing and plotting.
- excel_handler.py: Handles loading Excel files and saving predictions.
- data_preprocessing.py: Preprocesses Excel data, including column matching and grouping.
- gui_feedback.py: Provides a GUI for user feedback on model performance.
- logger.py: Logs training and testing feedback to CSV files.
- dashboard.py: Displays a GUI dashboard for visualizing model scores and loss history.
- predict_mode.py: Provides a GUI for loading Excel files and generating predictions.

How to Use
----------
### Prerequisites
- Data Files: Prepare Excel files (.xlsx) in the format shown in the Data Format section. Place training files in a `practice` folder and testing files in a `testing` folder.
- Directory Structure: Create the following folders in the project root:
  - practice/: For training data Excel files.
  - testing/: For testing data Excel files.
  - output/: Will store training prediction outputs (created automatically).
  - testing_output/: Will store testing prediction outputs (created automatically).
  - models/: Will store trained model weights (created automatically).
  - logs/: Will store feedback logs (created automatically).

### Running the Program
1. Launch the Program:
   python main.py
   This opens a GUI with options to:
   - Train Models: Process data in the `practice` folder.
   - Test Models: Process data in the `testing` folder.

### Processing Training Data
1. Prepare Data: Place Excel files in the `practice` folder. Ensure they follow the required format (see Data Format).
2. Run Training:
   - Select "Train Models" from the main GUI.
   - The program loads Excel files, groups data by Ionic Fluid, Nanoparticle, Size (nm), Concentration, and Shape, trains an ANN for each group, and saves predictions to the `output` folder.
   - A feedback GUI appears for each group, asking if the model's predictions are satisfactory. Respond with "Yes" or "No" to update the model's score.
   - If the average error exceeds 1.5%, the model retrains up to 5 times.
3. Outputs:
   - Predictions are saved as Excel files in the `output` folder (e.g., <original_filename>_prediction.xlsx).
   - Model weights are saved in the `models` folder.
   - Feedback logs are saved in logs/training_feedback_log.csv.

### Processing Testing Data
1. Prepare Data: Place Excel files in the `testing` folder.
2. Run Testing:
   - Select "Test Models" from the main GUI.
   - The program loads Excel files, groups data, loads existing models (if available), and generates predictions.
   - A feedback GUI appears for each group with errors below 1.5%.
3. Outputs:
   - Predictions are saved in the `testing_output` folder.
   - Feedback logs are saved in logs/testing_feedback_log.csv.

### Using Predict Mode
1. Run Predict Mode:
   python predict_mode.py
   This opens a GUI for single-file predictions.
2. Steps:
   - Click "Load Excel File" and select an Excel file.
   - Optionally enter a group ID (e.g., ('Ionic Fluid', 'Nanoparticle', 50.0, 10.0, 'Shape')) in the entry field. If blank, the group ID is inferred from the file.
   - Click "Run Prediction" to generate predictions.
   - Provide feedback via "Yes ✅" or "No ❌" buttons to update the model's score.
3. Outputs:
   - Predictions are appended to the input Excel file as a new column (Predicted Thermal Conductivity).
   - Feedback is logged in logs/training_feedback_log.csv.

### Viewing the Dashboard
1. Run Dashboard:
   python dashboard.py
   This opens a GUI to visualize model performance.
2. Steps:
   - Select one or more group IDs from the listbox.
   - Click "Update Plots" to display score over time and loss history plots.
   - Click "Export Summary to PDF" to save plots as a PDF.
3. Outputs:
   - Plots are displayed in the GUI.
   - Exported PDFs are saved to a user-specified location.

Data Format
-----------
Excel files must follow one of these formats:
- Training/Testing Data (e.g., Fox et. al (0.5_ wt Whisker) FINAL.xlsx):
  Columns: Base Liquid (or Ionic Fluid), Nanoparticle, Size (nm), Shape, Concentration (% wt), Temperature (deg. K), Thermal Conductivity (W/m*K).
  Example:
    Base Liquid,Nanoparticle,Size (nm),Shape,Concentration (% wt),Temperature (deg. K),Thermal Conductivity (W/m*K)
    [C4mmim][Tf2N],Aluminum Oxide,2-4,Whisker,0.5,303.1376908124907,0.126981208580517
  Notes:
    - Size (nm) can be a range (e.g., 2-4) or single value (e.g., 50).
    - Shape can be a specific value (e.g., Whisker) or - (unknown).
    - Header must be on row 2 (row 1 is ignored).

- Prediction Output (e.g., predictions__C4mmim__Tf2N__Aluminum_Oxide_3_0_0_5_Whisker.xlsx):
  Columns: Ionic Fluid, Nanoparticle, Size (nm), Concentration, Shape, Temperature, Thermal Conductivity, Predicted Thermal Conductivity.
  Generated automatically by the program.

Troubleshooting
---------------
- Missing Libraries: Ensure all libraries are installed (see Installation).
- File Not Found: Verify that `practice` and `testing` folders exist and contain Excel files.
- Column Mismatch: Check that Excel files have the required columns. The program uses fuzzy matching to handle variations (e.g., Base Liquid vs. Ionic Fluid).
- Model Errors: If predictions have high errors (>1.5%), try increasing max_attempts in controller.py or providing more training data.
- GUI Issues: Ensure tkinter is installed (included with Python). On Linux, you may need python3-tk.

Contributing
------------
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Submit a pull request with a description of your changes.
Please include tests and update this README if new dependencies or features are added.
