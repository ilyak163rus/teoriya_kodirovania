import numpy as np

# Генерация порождающей матрицы для (7, 4, 3)
def create_generator_matrix():
    identity_matrix = np.eye(4, dtype=int)
    parity_check_matrix = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]])
    generator_matrix = np.hstack((identity_matrix, parity_check_matrix))
    return generator_matrix

# Генерация матриц для (10, 5, 5)
def create_generator_matrix_5():
    identity_matrix = np.eye(5, dtype=int)
    parity_check_matrix_5 = np.array(
        [[1, 1, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [0, 1, 1, 1, 0],
         [1, 1, 0, 0, 1],
         [0, 0, 1, 1, 1]])
    generator_matrix_5 = np.hstack((identity_matrix, parity_check_matrix_5))
    return generator_matrix_5

# Генерация проверочной матрицы
def create_parity_check_matrix():
    parity_check_matrix = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]])
    identity_matrix = np.eye(3, dtype=int)
    check_matrix = np.hstack((parity_check_matrix.T, identity_matrix))
    return check_matrix

# Создание синдромов для ошибок
def create_syndrome_lookup(check_matrix):
    syndrome_dict = {}
    for i in range(check_matrix.shape[1]):
        error_vector = np.zeros(check_matrix.shape[1], dtype=int)
        error_vector[i] = 1
        syndrome = np.dot(check_matrix, error_vector) % 2
        syndrome_dict[tuple(syndrome)] = error_vector
    return syndrome_dict

# Формирование кодового слова
def form_codeword(data_bits, generator_matrix):
    return np.dot(data_bits, generator_matrix) % 2

# Внесение ошибок
def add_error(codeword, positions):
    for position in positions:
        codeword[position] ^= 1 # Инвертирование бита через ^
    return codeword

# Вычисление синдрома
def compute_syndrome(received_bits, check_matrix):
    return np.dot(check_matrix, received_bits) % 2

# Исправление ошибок на основе синдрома
def fix_error(received_bits, syndrome, syndrome_dict):
    if tuple(syndrome) in syndrome_dict:
        error_vector = syndrome_dict[tuple(syndrome)]
        corrected_bits = (received_bits + error_vector) % 2
        return corrected_bits
    return received_bits

# Внесение трёхкратной ошибки
def add_triple_error(codeword, positions):
    for position in positions:
        codeword[position] ^= 1
    return codeword

# Генерация проверочной матрицы
def create_parity_check_matrix_5(generator_matrix_5):
    parity_check_matrix_5 = generator_matrix_5[:, 5:]
    identity_matrix_5 = np.eye(5, dtype=int)
    check_matrix_5 = np.hstack((parity_check_matrix_5.T, identity_matrix_5))
    return check_matrix_5

# Генерация синдромов для двукратных ошибок
def create_double_error_syndromes(check_matrix):
    syndrome_dict = {}
    n = check_matrix.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            error_vector = np.zeros(n, dtype=int)
            error_vector[i] = 1
            error_vector[j] = 1
            syndrome = np.dot(check_matrix, error_vector) % 2
            syndrome_dict[tuple(syndrome)] = error_vector
    return syndrome_dict


generator_matrix = create_generator_matrix()
print("-------Часть 1---------")
print("Сгенерированная матрица:")
print(generator_matrix)

parity_check_matrix = create_parity_check_matrix()
print("\nПроверочная матрица:")
print(parity_check_matrix)

syndrome_dict = create_syndrome_lookup(parity_check_matrix)
print("\nСиндромы для ошибок:")
for syndrome, error in syndrome_dict.items():
    print(f"Синдром {syndrome}: Ошибка {error}")

data_bits = np.array([1, 0, 1, 1])
codeword = form_codeword(data_bits, generator_matrix)
print("\n>>>Сформированное кодовое слово:", codeword)

error_positions = [2]
received_bits = add_error(codeword.copy(), error_positions)
print("\nПолученное слово с ошибкой:", received_bits)

syndrome = compute_syndrome(received_bits, parity_check_matrix)
print("Вычисленный синдром:", syndrome)

corrected_bits = fix_error(received_bits, syndrome, syndrome_dict)
print("Исправленное слово:", corrected_bits)

double_error_positions = [1, 4]
received_bits_double = add_error(codeword.copy(), double_error_positions)
print("\nПолученное слово с двукратной ошибкой:", received_bits_double)

syndrome_double = compute_syndrome(received_bits_double, parity_check_matrix)
print("Синдром для двукратной ошибки:", syndrome_double)

corrected_double = fix_error(received_bits_double, syndrome_double, syndrome_dict)
print("Попытка исправления:", corrected_double)
print("Отличие от оригинала:", not np.array_equal(corrected_double, codeword))

generator_matrix_5 = create_generator_matrix_5()
print("\n\n-------Часть 2---------")
print("\n Сгенерированная матрица для (10, 5, 5) кода:")
print(generator_matrix_5)

parity_check_matrix_5 = create_parity_check_matrix_5(generator_matrix_5)
print("\nПроверочная матрица для (10, 5) кода:")
print(parity_check_matrix_5)

double_error_syndrome_dict = create_double_error_syndromes(parity_check_matrix_5)
print("\nСиндромы для двукратных ошибок:")
for syndrome, error in double_error_syndrome_dict.items():
    print(f"Синдром {syndrome}: Ошибка {error}")

data_bits_5 = np.array([1, 1, 1, 1, 0])
codeword_5 = form_codeword(data_bits_5, generator_matrix_5)
print("\n>>>Сформированное кодовое слово (d=5):", codeword_5)

error_position_5 = 9
received_bits_5 = add_error(codeword_5.copy(), [error_position_5])
print("\nПолученное слово с ошибкой:", received_bits_5)

syndrome_5 = compute_syndrome(received_bits_5, parity_check_matrix_5)
print("Вычисленный синдром:", syndrome_5)

corrected_bits_5 = fix_error(received_bits_5, syndrome_5, syndrome_dict)
print("Исправленное слово:", corrected_bits_5)

double_error_positions_5 = [0, 9]
received_bits_double_5 = add_error(codeword_5.copy(), double_error_positions_5)
print("\nПолученное слово с двукратной ошибкой:", received_bits_double_5)

syndrome_double_5 = compute_syndrome(received_bits_double_5, parity_check_matrix_5)
print("Синдром для двукратной ошибки:", syndrome_double_5)

corrected_double_5 = fix_error(received_bits_double_5, syndrome_double_5, double_error_syndrome_dict)
print("Попытка исправления:", corrected_double_5)
print("Отличие от оригинала:", not np.array_equal(corrected_double_5, codeword_5))

data_bits = np.array([1, 0, 1, 1])
codeword_triple = form_codeword(data_bits, generator_matrix)
print("\n>>>Сформированное кодовое слово:", codeword_triple)

triple_error_positions = [0, 2, 5]
received_bits_triple = add_triple_error(codeword_triple.copy(), triple_error_positions)
print("\nПолученное слово с трёхкратной ошибкой:", received_bits_triple)

syndrome_triple = compute_syndrome(received_bits_triple, parity_check_matrix)
print("Вычисленный синдром:", syndrome_triple)

corrected_triple = fix_error(received_bits_triple, syndrome_triple, syndrome_dict)
print("Попытка исправления:", corrected_triple)

print("Отличие от оригинала:", not np.array_equal(corrected_triple, codeword_triple))
