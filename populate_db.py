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
        ('Rice', 'Kharif (Monsoon)', 'Fácil', 'Ideal para climas cálidos y húmedos.'),
        ('Cotton', 'Kharif (Monsoon)', 'Difícil', 'Cultivo importante para la industria textil.'),
        ('Wheat', 'Rabi (Winter)', 'Medio', 'Uno de los principales granos básicos.'),
        ('Tomatoes', 'Rabi (Winter)', 'Fácil', 'Cultivo versátil usado en muchas cocinas.'),
        ('Maize', 'Kharif (Monsoon)', 'Medio', 'Conocido como maíz, es un cultivo cereal esencial.'),
        ('Barley', 'Rabi (Winter)', 'Fácil', 'Grano utilizado en la producción de cervezas.'),
        ('Sugarcane', 'Kharif (Monsoon)', 'Difícil', 'Fuente principal de azúcar en el mundo.'),
        ('Potatoes', 'Rabi (Winter)', 'Fácil', 'Tubérculo versátil y altamente nutritivo.'),
        ('Millets', 'Kharif (Monsoon)', 'Medio', 'Cereales pequeños ideales para climas secos.')
    ]
    for crop in crops:
        cursor.execute('INSERT INTO crops (name, season, difficulty, description) VALUES (?, ?, ?, ?)', crop)

    # Relación suelos-cultivos con compatibilidad
    soil_crop_relations = [
        (1, 1, 'Alta'),  # Alluvial Soil - Rice
        (1, 3, 'Alta'),  # Alluvial Soil - Wheat
        (1, 5, 'Media'), # Alluvial Soil - Maize
        (2, 2, 'Alta'),  # Black Soil - Cotton
        (2, 9, 'Media'), # Black Soil - Millets
        (3, 1, 'Media'), # Clay Soil - Rice
        (3, 8, 'Alta'),  # Clay Soil - Potatoes
        (4, 4, 'Alta'),  # Red Soil - Tomatoes
        (4, 7, 'Media'), # Red Soil - Sugarcane
        (4, 6, 'Baja')   # Red Soil - Barley
    ]
    for relation in soil_crop_relations:
        cursor.execute('INSERT INTO soil_crops (soil_id, crop_id, compatibility) VALUES (?, ?, ?)', relation)


    # Guardar cambios y cerrar conexión
    conn.commit()
    conn.close()

if __name__ == '__main__':
    populate_database()
    print("Database populated successfully!")
