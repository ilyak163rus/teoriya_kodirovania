import numpy as np
from itertools import combinations, product

# Генерация строки матрицы для заданного подмножества битов
def generate_row(vectors, subset):
    return [
        (np.prod([(x + 1) for i, x in enumerate(vector) if i in subset]) % 2)
        for vector in vectors
    ]

# Сортировка строк матрицы по степени мономов
def sort_rows(matrix, subset_lengths):
    def sort_key(row):
        # Определяем степень монома как индекс последней 1
        for idx in range(len(row) - 1, -1, -1):
            if row[idx] == 1:
                return idx
        return len(row)
    
    # Разделяем строки матрицы по длинам подмножеств и сортируем каждую часть
    sorted_matrix = []
    start = 0
    for length in subset_lengths:
        end = start + length
        sorted_matrix.extend(sorted(matrix[start:end], key=sort_key))
        start = end

    return np.array(sorted_matrix, dtype=int)

# Построение генераторной матрицы кода Рида-Маллера
def reed_muller_matrix(r, m):
    # Генерация всех векторов длины m
    bit_vectors = [list(map(int, f"{i:0{m}b}"[::-1])) for i in range(2 ** m)]
    
    # Формируем подмножества индексов и определяем их длины
    subsets, subset_lengths = [], []
    for i in range(r + 1):
        subset = list(combinations(range(m), i))
        subsets.extend(subset)
        subset_lengths.append(len(subset))
    
    # Генерируем строки для всех подмножеств
    matrix = np.array([generate_row(bit_vectors, subset) for subset in subsets], dtype=int)
    return sort_rows(matrix, subset_lengths)

# Генерация всех бинарных векторов заданной длины
def generate_binary_matrix(cols):
    return list(product([0, 1], repeat=cols))

# Вычисление значения функции f для заданного подмножества
def compute_f_value(vector, subset):
    return np.prod([(vector[idx] + 1) % 2 for idx in subset])

# Генерация вектора v для заданного подмножества
def generate_v_vector(subset, m):
    if not subset:
        return np.ones(2 ** m, dtype=int)  # Все единицы для пустого подмножества
    return [compute_f_value(vector, subset) for vector in generate_binary_matrix(m)]

# Генерация дополнения к заданному подмножеству
def generate_complement_set(subset, m):
    return [i for i in range(m) if i not in subset]

# Декодирование методом мажоритарного голосования
def major_decode(received_word, r, m):
    word = received_word.copy()  # Копия принятого слова
    max_weight = 2 ** (m - r - 1) - 1  # Порог веса для исправления ошибок
    decoded_word = np.zeros(2 ** m, dtype=int)  # Инициализация декодированного слова
    index = 0  # Текущий индекс в декодированном слове
    
    for i in range(r, -1, -1):  # Обрабатываем мономы от максимальной степени к нулевой
        for subset in combinations(range(m), i):
            ones_count = zeros_count = 0
            for t in generate_binary_matrix(len(subset)):
                complement = generate_complement_set(subset, m)
                v_vector = generate_v_vector(complement, m)
                c = np.dot(word, v_vector) % 2
                if c == 0:
                    zeros_count += 1
                else:
                    ones_count += 1
            
            # Исправляем ошибки на основе веса
            if zeros_count > max_weight:
                decoded_word[index] = 0
                index += 1
            elif ones_count > max_weight:
                decoded_word[index] = 1
                v_vector = generate_v_vector(subset, m)
                word = (word + v_vector) % 2
                index += 1
    
    return decoded_word

# Генерация кодового слова с ошибками
def generate_error_word(G, error_count):
    # Исходное сообщение
    message = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    # Кодирование сообщения
    encoded = message.dot(G) % 2
    # Генерация случайных ошибок
    error_indices = np.random.choice(len(encoded), size=error_count, replace=False)
    encoded[error_indices] = (encoded[error_indices] + 1) % 2
    return encoded

if __name__ == "__main__":
    r, m = 2, 4  # Параметры кода Рида-Маллера
    G = reed_muller_matrix(r, m)  # Генераторная матрица
    
    # Тест: однократная ошибка
    word_with_error = generate_error_word(G, 1)
    print("Слово с однократной ошибкой:", word_with_error)
    decoded = major_decode(word_with_error, r, m)
    if decoded is None:
        print("Необходима повторная отправка сообщения.")
    else:
        print("Исправленное слово:", decoded)
        result = decoded[:G.shape[0]].dot(G) % 2
        print("Результат проверки исправленного слова:", result)
    
    # Тест: двукратная ошибка
    word_with_error = generate_error_word(G, 2)
    print("\nСлово с двукратной ошибкой:", word_with_error)
    decoded = major_decode(word_with_error, r, m)
    if decoded is None:
        print("Необходима повторная отправка сообщения.")
    else:
        print("Исправленное слово:", decoded)
        result = decoded[:G.shape[0]].dot(G) % 2
        print("Результат проверки исправленного слова:", result)
