import numpy as np
import random

# Расширенный код Голея
B = np.array([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
])

def generate_matrices(B):
    """
    Генерация порождающей (G) и проверочной (H) матриц для расширенного кода Голея.

    Параметры:
        B (np.ndarray): Бинарная матрица, определяющая код Голея.

    Возвращает:
        tuple: Порождающая матрица G и проверочная матрица H.
    """
    G = np.hstack((np.eye(12, dtype=int), B))
    H = np.vstack((np.eye(12, dtype=int), B))
    return G, H

def introduce_errors(word, G, error_count):
    """
    Внесение ошибок в передаваемое кодовое слово.

    Параметры:
        word (np.ndarray): Исходное кодовое слово.
        G (np.ndarray): Порождающая матрица.
        error_count (int): Количество ошибок для внесения.

    Возвращает:
        np.ndarray: Кодовое слово с ошибками.
    """
    coded_word = word @ G % 2
    error_positions = random.sample(range(coded_word.shape[0]), error_count)
    error_vector = np.zeros(coded_word.shape[0], dtype=int)
    for index in error_positions:
        error_vector[index] = 1
    return (coded_word + error_vector) % 2

def detect_errors(received_word, H, B):
    """
    Обнаружение и локализация ошибок в принятом кодовом слове с использованием проверочной матрицы.

    Параметры:
        received_word (np.ndarray): Принятое кодовое слово с ошибками.
        H (np.ndarray): Проверочная матрица.
        B (np.ndarray): Матрица, определяющая код Голея.

    Возвращает:
        np.ndarray или None: Выявленный паттерн ошибок или None, если ошибки не поддаются исправлению.
    """
    syndrome = received_word @ H % 2
    if sum(syndrome) <= 3:
        return np.hstack((syndrome, np.zeros(len(syndrome), dtype=int)))
    for i, b_row in enumerate(B):
        temp_syndrome = (syndrome + b_row) % 2
        if sum(temp_syndrome) <= 2:
            mistake_index = np.zeros(len(syndrome), dtype=int)
            mistake_index[i] = 1
            return np.hstack((temp_syndrome, mistake_index))
    return None

def correct_errors(original_word, received_word, H, B, G):
    """
    Исправление ошибок в принятом кодовом слове.

    Параметры:
        original_word (np.ndarray): Исходное кодовое слово.
        received_word (np.ndarray): Принятое кодовое слово с ошибками.
        H (np.ndarray): Проверочная матрица.
        B (np.ndarray): Матрица, определяющая код Голея.
        G (np.ndarray): Порождающая матрица.
    """
    error_pattern = detect_errors(received_word, H, B)
    if error_pattern is None:
        print("Обнаружена ошибка, исправление невозможно.")
        return
    corrected_word = (received_word + error_pattern) % 2
    expected_word = original_word @ G % 2
    if not np.array_equal(expected_word, corrected_word):
        print("Ошибка декодирования!")

def generate_rm_generator_matrix(r, m):
    """
    Генерация порождающей матрицы для кода Рида-Маллера RM(r, m).

    Параметры:
        r (int): Порядок кода.
        m (int): Длина кода.

    Возвращает:
        np.ndarray: Порождающая матрица для кода Рида-Маллера.
    """
    if 0 < r < m:
        upper_left = generate_rm_generator_matrix(r, m - 1)
        lower_right = generate_rm_generator_matrix(r - 1, m - 1)
        return np.hstack([
            np.vstack([upper_left, np.zeros((len(lower_right), len(upper_left.T)), int)]),
            np.vstack([upper_left, lower_right])
        ])
    elif r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    elif r == m:
        upper = generate_rm_generator_matrix(m - 1, m)
        lower = np.zeros((1, 2 ** m), dtype=int)
        lower[0][-1] = 1
        return np.vstack([upper, lower])

def generate_rm_check_matrix(i, m):
    """
    Генерация проверочной матрицы для кода Рида-Маллера RM(r, m).

    Параметры:
        i (int): Текущий уровень построения по методу произведения Кронекера.
        m (int): Длина кода.

    Возвращает:
        np.ndarray: Проверочная матрица для RM(i, m).
    """
    H = np.array([[1, 1], [1, -1]])
    result = np.kron(np.eye(2 ** (m - i)), H)
    return np.kron(result, np.eye(2 ** (i - 1)))

def test_rm_code(word, G, error_count, m):
    """
    Исследование кода Рида-Маллера RM(r, m) в условиях ошибок.

    Параметры:
        word (np.ndarray): Исходное кодовое слово.
        G (np.ndarray): Порождающая матрица для RM(r, m).
        error_count (int): Количество вносимых ошибок.
        m (int): Длина кода.
    """
    received_word = introduce_errors(word, G, error_count)
    received_word = np.where(received_word == 0, -1, received_word)  # Заменяем 0 на -1 для декодирования
    w_t = [received_word @ generate_rm_check_matrix(1, m)]
    for i in range(2, m + 1):
        w_t.append(w_t[-1] @ generate_rm_check_matrix(i, m))

    # Поиск исправления с наибольшей вероятностью
    max_likelihood = w_t[0][0]
    index = -1
    for i in range(len(w_t)):
        for j in range(len(w_t[i])):
            if abs(w_t[i][j]) > abs(max_likelihood):
                index = j
                max_likelihood = w_t[i][j]

    if sum(abs(w_t[i][j]) == abs(max_likelihood) for i in range(len(w_t)) for j in range(len(w_t[i]))) > 1:
        print("Невозможно исправить ошибку: неоднозначное декодирование.\n")
        return

    # Исправление ошибок
    corrected_word = list(map(int, list(f"{index:0{m}b}")))
    corrected_word.append(1 if max_likelihood > 0 else 0)
    print(f"Исправленное сообщение: {np.array(corrected_word[::-1])}")

def investigate_golay_code():
    """
    Часть 1: Исследование расширенного кода Голея (24, 12, 8).
    """
    print("-------------------------------\nЧасть 1")
    G, H = generate_matrices(B)
    print(f"Порождающая матрица G:\n{G}\nПроверочная матрица H:\n{H}")

    original_word = np.array([i % 2 for i in range(len(G))])
    for error_count in range(5):  # Тестирование для 0-4 ошибок
        received_word = introduce_errors(original_word, G, error_count)
        correct_errors(original_word, received_word, H, B, G)
        print('')

def investigate_rm_codes():
    """
    Часть 2: Исследование кодов Рида-Маллера RM(1, 3) и RM(1, 4).
    """
    print("-------------------------------\nЧасть 2")

    # RM(1, 3)
    m = 3
    print(f"\nПорождающая матрица для RM(1, 3):\n{generate_rm_generator_matrix(1, m)}\n")
    word = np.array([i % 2 for i in range(len(generate_rm_generator_matrix(1, m)))])
    for error_count in range(1, 3):  # Тестирование для 1 и 2 ошибок
        test_rm_code(word, generate_rm_generator_matrix(1, m), error_count, m)

    # RM(1, 4)
    m = 4
    print(f"\nПорождающая матрица для RM(1, 4):\n{generate_rm_generator_matrix(1, m)}\n")
    word = np.array([i % 2 for i in range(len(generate_rm_generator_matrix(1, m)))])
    for error_count in range(1, 5):  # Тестирование для 1-4 ошибок
        test_rm_code(word, generate_rm_generator_matrix(1, m), error_count, m)

if __name__ == '__main__':
    investigate_golay_code()
    investigate_rm_codes()

