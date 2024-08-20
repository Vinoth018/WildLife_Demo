# WildLife Demo

This project is a demo application for wildlife detection using a pre-trained YOLO model. It includes features for detecting animal species, blacklisted vehicles, and vehicle numbers.

## Installation and Setup

Follow these steps to set up the environment and run the server:

### 1. Clone the Repository
```bash
git clone https://github.com/Vinoth018/WildLife_Demo.git
cd WildLife_Demo


2. Create and activate a virtual environment.
python -m venv myenv

.\myenv\Scripts\activate


3. Install the Required Dependencies
pip install -r requirements.txt



4. Fix OS-Related Errors (Optional)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu



5. Run the Application
python app.py



Database Schema
The application uses MySQL to store detected data. Use the following SQL commands to create the necessary tables:

1. Species Detection Table
CREATE TABLE species_detection (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detected_label VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL,
    sector VARCHAR(50) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);



2. Blacklisted Vehicles Table
CREATE TABLE blacklisted_vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_number VARCHAR(255) NOT NULL,
    sector VARCHAR(50) NOT NULL
);


                        
3. Vehicle Number Detection Table
CREATE TABLE vehicle_number_detection (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_number VARCHAR(255),
    sector VARCHAR(50),
    confidence FLOAT
);
