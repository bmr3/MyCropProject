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
    crops = [
        ('Rice', 'Kharif (Monsoon)', 'Fácil'),
        ('Cotton', 'Kharif (Monsoon)', 'Difícil'),
        ('Wheat', 'Rabi (Winter)', 'Medio'),
        ('Tomatoes', 'Rabi (Winter)', 'Fácil')
    ]
    for crop in crops:
        cursor.execute('INSERT INTO crops (name, season, difficulty) VALUES (?, ?, ?)', crop)

    # Relación suelos-cultivos
    soil_crop_relations = [
        (1, 1),  # Alluvial Soil - Rice
        (2, 2),  # Black Soil - Cotton
        (1, 3),  # Alluvial Soil - Wheat
        (4, 4)   # Red Soil - Tomatoes
    ]
    for relation in soil_crop_relations:
        cursor.execute('INSERT INTO soil_crops (soil_id, crop_id) VALUES (?, ?)', relation)

    # Guardar cambios y cerrar conexión
    conn.commit()
    conn.close()

if __name__ == '__main__':
    populate_database()
    print("Database populated successfully!")
