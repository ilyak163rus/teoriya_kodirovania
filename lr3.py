import numpy as np
np.random.seed(42)
# Генерация порождающей и проверочной матриц для кода Хэмминга
def generate_hamming_matrices(order):
    length = 2**order - 1  # Длина кодового слова
    info_length = length - order  # Длина информационного слова

    # Создаем проверочную матрицу
    parity_matrix = [list(map(int, bin(idx)[2:].zfill(order))) for idx in range(1, length + 1)]
    parity_matrix = np.array(parity_matrix).T

    # Создаем порождающую матрицу
    identity_matrix = np.eye(info_length, dtype=int)
    generator_matrix = np.hstack((identity_matrix, parity_matrix[:, :info_length].T))

    return generator_matrix, parity_matrix

# Создание таблицы синдромов для исправления одиночных ошибок
def create_syndrome_table(parity_matrix):
    length = parity_matrix.shape[1]
    syndrome_dict = {}
    for idx in range(length):
        error_vec = np.zeros(length, dtype=int)
        error_vec[idx] = 1
        syndrome = np.dot(error_vec, parity_matrix.T) % 2
        syndrome_dict[tuple(syndrome)] = error_vec
    return syndrome_dict

# Кодирование информационного слова с использованием матрицы G
def encode_message(info_word, generator_matrix):
    return np.dot(info_word, generator_matrix) % 2

# Декодирование кодового слова с использованием матрицы H и таблицы синдромов
def decode_message(received_word, parity_matrix, syndrome_dict):
    current_syndrome = np.dot(received_word, parity_matrix.T) % 2
    if tuple(current_syndrome) in syndrome_dict:
        error_vec = syndrome_dict[tuple(current_syndrome)]
        corrected_word = (received_word + error_vec) % 2
        return corrected_word, True
    return received_word, False

# Исследование ошибок для кодов Хэмминга
def investigate_hamming_errors(order):
    generator_matrix, parity_matrix = generate_hamming_matrices(order)
    syndrome_dict = create_syndrome_table(parity_matrix)

    info_word = np.random.randint(0, 2, size=generator_matrix.shape[0])
    encoded_word = encode_message(info_word, generator_matrix)

    print(f"\nПроверка кода Хэмминга для r = {order}")
    length = len(encoded_word)

    # Однократная ошибка
    error_vec = np.zeros(length, dtype=int)
    if length > 1:
        error_vec[1] = 1
        received_with_error = (encoded_word + error_vec) % 2
        corrected_word, success = decode_message(received_with_error, parity_matrix, syndrome_dict)
        print(f"Получено слово с однократной ошибкой: {received_with_error}, Исправленное слово: {corrected_word}, Было исправлено: {'Да' if success else 'Нет'}")

    # Двукратная ошибка
    if length > 3:
        error_vec[3] = 1
        received_with_error = (encoded_word + error_vec) % 2
        corrected_word, success = decode_message(received_with_error, parity_matrix, syndrome_dict)
        print(f"Получено слово с двукратной ошибкой: {received_with_error}, Исправленное слово: {corrected_word}, Было исправлено: {'Да' if success else 'Нет'}")

    # Трёхкратная ошибка
    if length > 5:
        error_vec[5] = 1
        received_with_error = (encoded_word + error_vec) % 2
        corrected_word, success = decode_message(received_with_error, parity_matrix, syndrome_dict)
        print(f"Получено слово с трёхкратной ошибкой: {received_with_error}, Исправленное слово: {corrected_word}, Было исправлено: {'Да' if success else 'Нет'}")

# Запуск исследования для значений r = 2, 3, 4
for order in [2, 3, 4]:
    investigate_hamming_errors(order)

# Генерация порождающей и проверочной матриц для расширенного кода Хэмминга
def generate_extended_hamming_matrices(order):
    generator_matrix, parity_matrix = generate_hamming_matrices(order)
    extended_length = 2**order

    # Добавляем столбец для расширения матриц
    extended_generator = np.hstack((generator_matrix, np.ones((generator_matrix.shape[0], 1), dtype=int)))
    extended_parity = np.hstack((parity_matrix, np.ones((parity_matrix.shape[0], 1), dtype=int)))

    return extended_generator, extended_parity

# Исследование расширенного кода Хэмминга
def investigate_extended_hamming(order):
    extended_generator, extended_parity = generate_extended_hamming_matrices(order)
    extended_syndrome_dict = create_syndrome_table(extended_parity)

    info_word = np.random.randint(0, 2, size=extended_generator.shape[0])
    encoded_word = encode_message(info_word, extended_generator)

    print(f"\nСиндромная таблица для расширенного кода Хэмминга (r = {order}):")
    for syndrome, error_vec in extended_syndrome_dict.items():
        print(f"Синдром {syndrome} -> Ошибка {error_vec}")

    print("\nПроверка кратных ошибок:")
    length = len(encoded_word)

    # Однократная ошибка
    error_vec = np.zeros(length, dtype=int)
    if length > 1:
        error_vec[1] = 1
        received_with_error = (encoded_word + error_vec) % 2
        corrected_word, success = decode_message(received_with_error, extended_parity, extended_syndrome_dict)
        print(f"Однократная ошибка: {received_with_error}, Исправление: {'Да' if success else 'Нет'}")

    # Двукратная ошибка
    if length > 3:
        error_vec[3] = 1
        received_with_error = (encoded_word + error_vec) % 2
        corrected_word, success = decode_message(received_with_error, extended_parity, extended_syndrome_dict)
        print(f"Двукратная ошибка: {received_with_error}, Исправление: {'Да' if success else 'Нет'}")

    # Трёхкратная ошибка
    if length > 5:
        error_vec[5] = 1
        received_with_error = (encoded_word + error_vec) % 2
        corrected_word, success = decode_message(received_with_error, extended_parity, extended_syndrome_dict)
        print(f"Трёхкратная ошибка: {received_with_error}, Исправление: {'Да' if success else 'Нет'}")

    # Четырёхкратная ошибка
    if length > 6:
        error_vec[6] = 1
        received_with_error = (encoded_word + error_vec) % 2
        corrected_word, success = decode_message(received_with_error, extended_parity, extended_syndrome_dict)
        print(f"Четырёхкратная ошибка: {received_with_error}, Исправление: {'Да' if success else 'Нет'}")

# Запуск исследования расширенного кода для значений r = 2, 3, 4
for order in [2, 3, 4]:
    investigate_extended_hamming(order)
