import sqlite3

def populate_database():
    # Conectar a la base de datos
    conn = sqlite3.connect('soil_crops.db')
    cursor = conn.cursor()

    # Insertar tipos de suelos
    soils = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']
    for soil in soils:
        cursor.execute('INSERT OR IGNORE INTO soils (name) VALUES (?)', (soil,))

    # Insertar cultivos
# Insertar cultivos con descripciones
    crops = [
    # Nombre, Estación, Dificultad, Descripción
    ('Rice', 'Kharif (Monsoon)', 'Easy', 'Ideal for warm and humid climates.'),
    ('Cotton', 'Kharif (Monsoon)', 'Hard', 'A key crop for the textile industry.'),
    ('Wheat', 'Rabi (Winter)', 'Medium', 'One of the world’s staple grains.'),
    ('Tomatoes', 'Rabi (Winter)', 'Easy', 'Versatile crop used in many cuisines.'),
    ('Maize', 'Kharif (Monsoon)', 'Medium', 'Also known as corn, a key cereal crop.'),
    ('Barley', 'Rabi (Winter)', 'Easy', 'A grain used in brewing beer.'),
    ('Sugarcane', 'Kharif (Monsoon)', 'Hard', 'Primary source of sugar globally.'),
    ('Potatoes', 'Rabi (Winter)', 'Easy', 'Versatile and highly nutritious tuber.'),
    ('Millets', 'Kharif (Monsoon)', 'Medium', 'Small cereals ideal for arid climates.'),
    ('Groundnuts', 'Kharif (Monsoon)', 'Easy', 'Also known as peanuts, rich in oil.'),
    ('Soybeans', 'Kharif (Monsoon)', 'Medium', 'High-protein crop used in various industries.'),
    ('Chickpeas', 'Rabi (Winter)', 'Easy', 'Nutrient-rich legumes ideal for dry climates.'),
    ('Carrots', 'Rabi (Winter)', 'Easy', 'Root vegetable known for its nutritional value.'),
    ('Onions', 'Rabi (Winter)', 'Medium', 'Widely used in cuisines worldwide.'),
    ('Jowar (Sorghum)', 'Kharif (Monsoon)', 'Medium', 'Cereal crop resistant to drought.'),
    ('Sunflower', 'Kharif (Monsoon)', 'Medium', 'Valuable for oil production.'),
    ('Peas', 'Rabi (Winter)', 'Easy', 'Rich in protein and used in many dishes.'),
    ('Lentils', 'Rabi (Winter)', 'Easy', 'Staple legume known for its versatility.')
]
    for crop in crops:
        cursor.execute('INSERT INTO crops (name, season, difficulty, description) VALUES (?, ?, ?, ?)', crop)

    # Relación suelos-cultivos con compatibilidad
    soil_crop_relations = [
        # (soil_id, crop_id, compatibility)
        # Alluvial Soil (ID 1)
        (1, 1, 'High'),    # Rice
        (1, 3, 'High'),    # Wheat
        (1, 5, 'Medium'),  # Maize
        (1, 10, 'Medium'), # Groundnuts
        (1, 18, 'Low'),    # Lentils

        # Black Soil (ID 2)
        (2, 2, 'High'),    # Cotton
        (2, 9, 'Medium'),  # Millets
        (2, 13, 'Medium'), # Chickpeas
        (2, 15, 'Low'),    # Onions
        (2, 16, 'Low'),    # Jowar (Sorghum)

        # Clay Soil (ID 3)
        (3, 8, 'High'),    # Potatoes
        (3, 1, 'Medium'),  # Rice
        (3, 4, 'Medium'),  # Tomatoes
        (3, 12, 'Medium'), # Soybeans
        (3, 14, 'Low'),    # Carrots

        # Red Soil (ID 4)
        (4, 4, 'High'),    # Tomatoes
        (4, 7, 'Medium'),  # Sugarcane
        (4, 9, 'Medium'),  # Millets
        (4, 6, 'Low'),     # Barley
        (4, 11, 'Low')     # Sunflower
    ]
    
    for relation in soil_crop_relations:
        cursor.execute('INSERT INTO soil_crops (soil_id, crop_id, compatibility) VALUES (?, ?, ?)', relation)


    # Guardar cambios y cerrar conexión
    conn.commit()
    conn.close()

if __name__ == '__main__':
    populate_database()
    print("Database populated successfully!")
