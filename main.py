import numpy as np
from itertools import combinations

# Функция для приведения матрицы к ступенчатому виду
def to_row_echelon_form(mat):
    mat = np.array(mat, dtype=int)
    rows, cols = mat.shape
    lead = 0
    for r in range(rows):
        if lead >= cols:
            return mat
        i = r
        while mat[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    return mat
        mat[[i, r]] = mat[[r, i]]
        for i in range(r + 1, rows):
            if mat[i, lead] != 0:
                mat[i] = (mat[i] + mat[r]) % 2
        lead += 1
    return mat

print("Задание 1.1")
matrix = [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 0, 1]
]
result = to_row_echelon_form(matrix)
print(result.astype(int))

# Функция для приведения матрицы к приведенному ступенчатому виду
def to_reduced_row_echelon_form(mat):
    mat = to_row_echelon_form(mat)
    rows, cols = mat.shape

    for r in range(rows - 1, -1, -1):
        lead = np.argmax(mat[r] != 0)
        if mat[r, lead] != 0:
            for i in range(r - 1, -1, -1):
                if mat[i, lead] != 0:
                    mat[i] = (mat[i] + mat[r]) % 2
    while not any(mat[rows - 1]):
        mat = mat[:-1, :]
        rows -= 1
    return mat

print("Задание 1.2")
result = to_reduced_row_echelon_form(matrix)
print(result.astype(int))

# Класс для работы с линейным кодом
class LinearCode:
    def __init__(self, vectors):
        self.S = np.array(vectors, dtype=int)
        self.G = self.to_rref(self.S)
        self.n = self.G.shape[1]
        self.k = self.G.shape[0]
        self.G_star = self.modify_G(self.G)
        self.leading_cols = self.find_leading_cols(self.G_star)
        self.X = self.build_X(self.G_star, self.leading_cols)
        self.H = self.create_check_matrix(self.X, self.leading_cols)

    # Приведение матрицы к приведенному ступенчатому виду
    def to_rref(self, mat):
        return to_reduced_row_echelon_form(mat)

    # Модификация порождающей матрицы G
    def modify_G(self, G):
        rows, cols = G.shape
        G_star = G.copy()
        for r in range(rows):
            lead = np.nonzero(G_star[r])[0][0] if np.any(G_star[r]) else -1
            if lead != -1:
                for i in range(rows):
                    if i != r and G_star[i, lead] == 1:
                        G_star[i] = (G_star[i] - G_star[r]) % 2
        return G_star

    # Нахождение ведущих столбцов в матрице G*
    def find_leading_cols(self, G_star):
        leads = []
        for r in range(G_star.shape[0]):
            for c in range(G_star.shape[1]):
                if G_star[r, c] == 1 and c not in leads:
                    leads.append(c)
                    break
        return leads

    # Построение сокращенной матрицы X
    def build_X(self, G_star, lead):
        non_lead_cols = [c for c in range(G_star.shape[1]) if c not in lead]
        return G_star[:, non_lead_cols]

    # Создание проверочной матрицы H
    def create_check_matrix(self, mat, lead):
        id_matrix = np.eye(mat.shape[1], dtype=int)
        result_matrix = np.zeros((mat.shape[0] + id_matrix.shape[0], mat.shape[1]), dtype=int)
        idx_1, idx_2 = 0, 0

        for i in range(result_matrix.shape[0]):
            if i in lead:
                result_matrix[i] = mat[idx_1]
                idx_1 += 1
            else:
                result_matrix[i] = id_matrix[idx_2]
                idx_2 += 1

        return result_matrix

    # Генерация кодовых слов
    def generate_codewords(self, from_G=True):
        rows = self.G.shape[0]
        codewords = set()

        if from_G:
            for r in range(1, rows + 1):
                for comb in combinations(range(rows), r):
                    codeword = np.bitwise_xor.reduce(self.G[list(comb)], axis=0)
                    codewords.add(tuple(codeword))
            codewords.add(tuple(np.zeros(self.G.shape[1], dtype=int)))
        else:
            k = self.G.shape[0]
            for i in range(2 ** k):
                binary_word = np.array(list(np.binary_repr(i, k)), dtype=int)
                codeword = np.dot(binary_word, self.G) % 2
                codewords.add(tuple(codeword))

        return np.array(list(codewords))

    # Вычисление минимального расстояния кода
    def min_distance(self):
        codewords = self.generate_codewords()
        d_min = float('inf')
        for i in range(len(codewords)):
            for j in range(i + 1, len(codewords)):
                d = np.sum(np.array(codewords[i]) != np.array(codewords[j]))
                if d < d_min:
                    d_min = d
        return d_min, d_min // 2

    # Проверка кодового слова с помощью проверочной матрицы H
    def check_codeword(self, codeword):
        return np.dot(codeword, self.H) % 2

S = [
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
]

linear_code = LinearCode(S)
print("1.3.1: Порождающая матрица в ступенчатом виде:")
print(linear_code.G)

print("\n1.3.2: Определение n и k:")
print("n =", linear_code.n)
print("k =", linear_code.k)

print("\n1.3.3\n Шаг 1: Приведённая порождающая матрица G*:")
print(linear_code.G_star)

print("\nШаг 2: Ведущие столбцы:")
print(linear_code.leading_cols)

print("\nШаг 3: Сокращённая матрица X:")
print(linear_code.X)

print("\nШаг 4: Проверочная матрица H:")
print(linear_code.H)

print("\nЗадание 1.4: Генерация кодовых слов:")
codewords_1 = linear_code.generate_codewords(from_G=True)
codewords_2 = linear_code.generate_codewords(from_G=False)

# Проверяем списки кодовых слов на совпадение
if set(map(tuple, codewords_1)) != set(map(tuple, codewords_2)):
    raise AssertionError("Наборы кодовых слов не совпадают!")

# Проверка кодовых слов с помощью матрицы H
for codeword in codewords_2:
    result = linear_code.check_codeword(codeword)
    if not np.all(result == 0):
        raise AssertionError(f"Ошибка: кодовое слово {codeword} не прошло проверку матрицей H")

print("\nКодовые слова (1):\n", codewords_1)
print("\nКодовые слова (2):\n", codewords_2)

print("\n1.5: Вычисление минимального расстояния кода:")
min_distance, t = linear_code.min_distance()
print("Минимальное расстояние d =", min_distance)
print("Кратность обнаруживаемых ошибок t =", t)

v = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
e1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
v_e1 = (v + e1) % 2
print(f"e1 = {e1}")
print('v + e1 = ', v_e1)
print('(v + e1)@H =', linear_code.check_codeword(v_e1), "- error")

