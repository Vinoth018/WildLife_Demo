git clone https://github.com/Vinoth018/WildLife_Demo.git

cd WildLife_Demo

create env -->  python -m venv myenv
activate --> ./myenv/Scripts/activate

pip install -r requirements.txt

Run the server --> python app.py



fix os error  --> pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu


Database tables
1. CREATE TABLE species_detection (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detected_label VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL,
    sector VARCHAR(50) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

3. CREATE TABLE blacklisted_vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_number VARCHAR(255) NOT NULL,
    sector VARCHAR(50) NOT NULL
);

4. CREATE TABLE vehicle_number_detection (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_number VARCHAR(255),
    sector VARCHAR(50),
    confidence FLOAT
);
