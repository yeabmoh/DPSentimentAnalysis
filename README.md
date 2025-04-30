# DPSentimentAnalysis

DPSentimentAnalysis is a sentiment analysis project that uses BERT embeddings and logistic regression models (including a DP-SGD variant) to classify tweets into positive and negative sentiments. The project integrates with a Qdrant vector database for efficient storage and retrieval of embeddings.

---

## Table of Contents
1. [Project Setup](#project-setup)
2. [Running the Project](#running-the-project)
3. [Docker Setup for Qdrant](#docker-setup-for-qdrant)
4. [Folder Structure](#folder-structure)
5. [Key Features](#key-features)
6. [Troubleshooting](#troubleshooting)

---

## Project Setup

### Prerequisites
1. **Python**: Ensure Python 3.8+ is installed.
2. **Docker**: Install Docker to run the Qdrant database.
3. **Dependencies**: Install the required Python packages.

### Installation Steps
1. Clone the repository:
```bash
   git clone https://github.com/yeabmoh/DPSentimentAnalysis.git
   cd DPSentimentAnalysis 
```
2. Createt a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Create the `qdrant_data` folder:
```bash
   mkdir qdrant_data
```

## Running the Project

### Step 1: Start Qdrant
Start the Qdrant database using Docker:
```bash
   docker run -d -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant
   docker start qdrant
```
* `-p 6333:6333`: Maps port 6333 on your machine to Qdrant's port.
* `-v $(pwd)/qdrant_data:/qdrant/storage`: Mounts the qdrant_data folder for persistent storage. Make sure `$(pwd)/qdrant_data` is an absolute path to the qdrant_data foler you just created in the home repository with the `mkdir` command
* `--name qdrant`: Names the container qdrant.

### Step 2: Run the Sentiment Analysis Pipeline
Run the main script:
```bash
python run.py --model logistic
```

* Options:
  * `--model logistic`: Use the standard logistic regression model
  * `--model dp_logistic`: Use the DP-SGD logistic regression model

The script will:
1. Check if the qdrant_data folder is empty.
2. If empty, preprocess the dataset and store embeddings in Qdrant.
3. Load train and test data from Qdrant.
4. Train and evaluate the specified model.

## Docker Setup for Qdrant

### Starting Qdrant
To start the Qdrant container:
```bash
docker start qdrant
```

### Stopping Qdrant
To stop the Qdrant container:
```bash
docker stop qdrant
```

### Removing Qdrant Data
To clear the `qdrant_data` container:
```bash
rm -rf qdrant_data/*
```

### Folder Structure
```bash
DPSentimentAnalysis/
├── data/
│   ├── qdrant_db/          # Qdrant client and utility functions
│   ├── scripts/            # Scripts for preprocessing and loading data
│   ├── [settings.py](http://_vscodecontentref_/2)         # Configuration settings
├── models/
│   ├── [logistic.py](http://_vscodecontentref_/3)         # Logistic regression model
│   ├── [dp_logistic.py](http://_vscodecontentref_/4)      # DP-SGD logistic regression model
├── qdrant_data/            # Qdrant database storage (ignored by Git)
├── [requirements.txt](http://_vscodecontentref_/5)        # Python dependencies
├── [run.py](http://_vscodecontentref_/6)                  # Main script to run the pipeline
├── [README.md](http://_vscodecontentref_/7)               # Project documentation
```

### Key Features
1. BERT Tokenization:
  * Preprocesses tweets using the bert-base-uncased tokenizer.
  * Stores embeddings in Qdrant for efficient retrieval.
2. Logistic Regression Models:
  * Standard logistic regression (logistic).
  * Differentially private logistic regression (dp_logistic).
3. Qdrant Integration:
  * Uses Qdrant as a vector database for storing and retrieving embeddings.
Modular Design:
* Preprocessing, data loading, and model training are modular and reusable.

## Troubleshooting

### Issue: qdrant_data is empty
* Ensure the Qdrant container is running:
```bash
    docker start qdrant
```
* If the folder is still empty, rerun the preprocessing step:
```bash
    python run.py --model logistic
```

### Issue: `X_train` contains `nan` values
* Check the `bert_preprocessing_script.py` for issues with tokenization or vector creation.
* Ensure the `PointStruct` objects are correctly created and inserted into Qdrant.

### Issue: Qdrant container not starting
* Check if the container is already running:
```bash
    docker ps
```
If not, start it:
```bash
    docker start qdrant
```

## Future Improvements
* Add support for additional models.
* Implement advanced evaluation metrics.
* Optimize Qdrant queries for large datasets.

## License
License
This project is licensed under the MIT License. See the `LICENSE` file for details.


---

### Key Sections:
1. **Project Setup**:
   - Explains prerequisites, installation steps, and creating the [qdrant_data](http://_vscodecontentref_/8) folder.

2. **Running the Project**:
   - Details the steps to start Qdrant and run the pipeline.

3. **Docker Setup for Qdrant**:
   - Provides commands to manage the Qdrant container.

4. **Folder Structure**:
   - Describes the organization of the project.

5. **Troubleshooting**:
   - Addresses common issues like empty [qdrant_data](http://_vscodecontentref_/9) or `nan` values in [X_train](http://_vscodecontentref_/10).

6. **Future Improvements**:
   - Suggests potential enhancements for the project.

Let me know if you need further adjustments!