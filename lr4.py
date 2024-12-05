import numpy as np
from itertools import product


def golay_matrix_rmrix():
    # Генерация матриц для кода Голая
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
    G = np.hstack((np.eye(len(B), dtype=int), B))  # Генерация матрицы кодирования
    H = np.vstack((np.eye(len(B), dtype=int), B))  # Генерация матрицы контроля
    return B, G, H


def make_errors(code, n_errors):
    # Генерация ошибок в коде
    modified_code = code.copy()
    error_positions = np.random.choice(len(code), n_errors, replace=False)
    for pos in error_positions:
        modified_code[pos] = 1 - modified_code[pos]  # Переворачиваем биты в случайных позициях
    return modified_code


def decode(received, H):
    # Декодирование полученного кода с помощью матрицы контроля
    return np.dot(received, H) % 2


def golay_error_correction(G, H):
    # Основная функция для исправления ошибок в коде Голая
    msg = np.random.randint(2, size=12)  # Генерация случайного сообщения
    code = np.dot(msg, G) % 2  # Кодирование сообщения
    print(f"Исходное сообщение: {msg}")
    print(f"Сгенерированный код: {code}")

    # Генерация всех возможных слов для декодирования
    words = np.array(list(product([0, 1], repeat=12)))
    word_dict = {np.array_str((word @ G) % 2): word for word in words}

    for num_errors in range(1, 5):  # Пробуем количество ошибок от 1 до 4
        print(f"\nКоличество ошибок: {num_errors}")
        received_with_errors = make_errors(code, num_errors)  # Вставляем ошибки в код
        print(f"Получено с ошибками: {received_with_errors}")
        syndrome = decode(received_with_errors, H)  # Декодирование с полученными ошибками
        print(f"Синдром: {syndrome}")

        # Проверка ошибок и попытка их исправления
        error_vector = np.array([])
        if sum(syndrome) > 3:
            for i in range(B.shape[0]):
                if sum((syndrome + B[i]) % 2) <= 2:
                    e_i = np.zeros(12, dtype=int)
                    e_i[i] = 1
                    error_vector = np.hstack(((syndrome + B[i]) % 2, e_i))
                    break
        else:
            error_vector = np.hstack((syndrome, np.zeros(len(B), dtype=int)))

        if error_vector.size == 0:
            syndrome2 = (syndrome @ B) % 2
            if sum(syndrome2) > 3:
                for i in range(B.shape[0]):
                    if sum((syndrome2 + B[i]) % 2) <= 2:
                        e_i = np.zeros(12, dtype=int)
                        e_i[i] = 1
                        error_vector = np.hstack((e_i, (syndrome2 + B[i]) % 2))
                        break
            else:
                error_vector = np.hstack((np.zeros(12, dtype=int), syndrome2))

        if error_vector.size > 0:
            print(f"Декодированное сообщение: {word_dict.get(np.array_str((received_with_errors + error_vector) % 2), 'Ошибка: не удается исправить')}")
        else:
            print("Ошибка обнаружена, не удалось исправить")


def rm_g(r, m):
    # Рекурсивная функция для генерации матрицы Рида-Мюллера
    if r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    if r == m:
        return np.vstack((rm_g(m - 1, m), np.array([0] * (2 ** m - 1) + [1])))

    matrix_rm = rm_g(r, m - 1)
    matrix_rm2 = rm_g(r - 1, m - 1)
    return np.vstack((np.hstack((matrix_rm, matrix_rm)), np.hstack((np.zeros((matrix_rm2.shape[0], matrix_rm.shape[1]), dtype=int), matrix_rm2))))


def decode_word(w, Hs, k, G_key_words):
    # Декодирование слова с помощью ключевых слов
    pred = []
    min_dist = np.inf
    for element in G_key_words:
        dist = sum((w + element) % 2)
        if dist < min_dist:
            pred = [element]
            min_dist = dist
        elif dist == min_dist:
            pred.append(element)

    if len(pred) == 1:
        w = pred[0]
        w_copy = w.copy()
        w_copy[w_copy == 0] = -1

        for i in range(len(Hs)):
            w_copy = np.dot(w_copy, Hs[i])

        w_copy_abs = np.abs(w_copy)
        j = np.argmax(w_copy_abs)
        bin_j = bin(j)[2:].zfill(m)

        fake_message = np.array([1 if w_copy[j] > 0 else 0] + list(map(int, bin_j[::-1])), dtype=int)
        print(f"Исходное сообщение: {k}")
        print(f"Декодированное слово: {fake_message}")
    else:
        print("Ошибка обнаружена, не удается исправить")


# Пример работы функции с параметрами r и m
r, m = 1, 3
Hs = [np.kron(np.kron(np.eye(2**(m - i), dtype=int), np.array([[1, 1], [1, -1]])), np.eye(2**(i - 1), dtype=int)) for i in range(1, m + 1)]

key_words = np.array(list(product([0, 1], repeat=4)))
G_key_words = np.array([(key_word @ rm_g(r, m)) % 2 for key_word in key_words])
k = np.array([1, 1, 0, 0])
w = (k @ rm_g(r, m)) % 2
w[0] = (w[0] + 1) % 2
decode_word(w, Hs, k, G_key_words)

# Изменение k и w для следующего теста
k = np.array([1, 1, 0, 0])
w = (k @ rm_g(r, m)) % 2
w[0] = (w[0] + 1) % 2
w[1] = (w[1] + 1) % 2
decode_word(w, Hs, k, G_key_words)

# То же самое для других значений r, m...
