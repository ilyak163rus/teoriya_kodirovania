import numpy as np

# Приведение к ступенчатому виду (REF)
def REF(matrix):
    matrix = np.array(matrix, dtype=bool)
    rows, cols = matrix.shape
    lead = 0

    for r in range(rows):
        if lead >= cols:
            return matrix
        i = r
        while not matrix[i, lead]:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    return matrix
        matrix[[i, r]] = matrix[[r, i]]
        for i in range(r+1, rows):
            if matrix[i, lead]:
                matrix[i] = np.logical_xor(matrix[i], matrix[r])
        lead += 1
    matrix = matrix[~np.all(matrix == 0, axis=1)]
    return matrix

print("Задание 1.1")
matrix = [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 0, 1]
]
result = REF(matrix)
print(result.astype(int))

# Приведение к приведенному ступенчатому виду (RREF)
def RREF(matrix):
    matrix = np.array(matrix, dtype=float)
    rows, cols = matrix.shape
    lead = 0
    for r in range(rows):
        if lead >= cols:
            return matrix
        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    return matrix
        matrix[[i, r]] = matrix[[r, i]]
        lv = matrix[r, lead]
        matrix[r] = matrix[r] / lv
        for i in range(rows):
            if i != r:
                lv = matrix[i, lead]
                matrix[i] = matrix[i] - lv * matrix[r]
        lead += 1
    return matrix

print("Задание 1.2")
result = RREF(matrix)
print(result.astype(int))

# Класс линейного кода
class LinearCode:
    def __init__(self, vectors):
        self.S = np.array(vectors)
        self.G = self.row_echelon_form(self.S)
        self.n = self.G.shape[1]
        self.k = self.G.shape[0]  # Количество строк в G
        self.G_star = self.transform_G(self.G)
        self.lead = self.get_leading_columns(self.G_star)
        self.X = self.form_X(self.G_star, self.lead)
        self.H = self.form_check_matrix(self.X, self.lead)

    def row_echelon_form(self, matrix):
        matrix = matrix.copy()
        rows, cols = matrix.shape
        lead = 0
        for r in range(rows):
            if lead >= cols:
                return matrix
            i = r
            while matrix[i, lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        return matrix
            matrix[[r, i]] = matrix[[i, r]]
            for i in range(r + 1, rows):
                if matrix[i, lead] == 1:
                    matrix[i] = (matrix[i] - matrix[r]) % 2
            lead += 1
        return matrix

    def count_nonzero_rows(self, matrix):
        return np.count_nonzero(~np.all(matrix == 0, axis=1))

    def transform_G(self, G):
        rows, cols = G.shape
        G_star = G.copy()
        for r in range(rows):
            lead = np.nonzero(G_star[r])[0][0] if np.any(G_star[r]) else -1
            if lead != -1:
                for i in range(rows):
                    if i != r and G_star[i, lead] == 1:
                        G_star[i] = (G_star[i] - G_star[r]) % 2
        return G_star

    def get_leading_columns(self, G_star):
        lead = []
        for r in range(G_star.shape[0]):
            for c in range(G_star.shape[1]):
                if G_star[r, c] == 1 and c not in lead:
                    lead.append(c)
                    break
        return lead

    def form_X(self, G_star, lead):
        non_lead_cols = [c for c in range(G_star.shape[1]) if c not in lead]
        return G_star[:, non_lead_cols]

    def form_check_matrix(self, X, lead):
        rows, cols = X.shape
        num_leads = len(lead)
        
        if rows + num_leads != self.n:
            raise ValueError("Размеры матриц не совпадают для формирования H.")
        
        H = np.zeros((num_leads + rows, self.n), dtype=int)
        
        for i, l in enumerate(lead):
            H[i, l] = 1
        
        H[num_leads:, [c for c in range(self.n) if c not in lead]] = X
    
        return H

    def generate_codewords_from_G(self):
        k = self.G.shape[0]  # Количество строк в G
        codewords = set()
        for i in range(2 ** k):
            binary_vector = np.array([int(x) for x in bin(i)[2:].zfill(k)])
            codeword = (np.dot(binary_vector, self.G) % 2).tolist()
            codewords.add(tuple(codeword))
        return [list(cw) for cw in codewords]

    def generate_codewords_from_binary(self):
        k = self.G.shape[0]  # Количество строк в G
        codewords = set()
        for i in range(2 ** k):
            binary_vector = np.array([int(x) for x in bin(i)[2:].zfill(k)])
            codeword = (np.dot(binary_vector, self.G) % 2).tolist()
            codewords.add(tuple(codeword))
        return [list(cw) for cw in codewords]

    def calculate_min_distance(self):
        codewords = self.generate_codewords_from_G()
        min_distance = float('inf')
        for i in range(len(codewords)):
            for j in range(i + 1, len(codewords)):
                distance = np.sum(np.array(codewords[i]) != np.array(codewords[j]))
                if distance < min_distance:
                    min_distance = distance
        return min_distance, min_distance // 2

    def print_codewords_with_input(self):
        print("Input:")
        print("G =")
        print(self.G)
        print("H =")
        print(self.H)
        print("Result:")
        for i in range(2 ** self.k):
            u = np.array([int(x) for x in bin(i)[2:].zfill(self.k)])
            print("u =", u)
            v = (np.dot(u, self.G) % 2)
            print("v = u@G =", v)
            v_H = (np.dot(v, self.H.T) % 2)
            print("v@H =", v_H)
            print()  # Пустая строка для разделения результатов

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
print(linear_code.lead)

print("\nШаг 3: Сокращённая матрица X:")
print(linear_code.X)

print("\nШаг 4: Проверочная матрица H:")
print(linear_code.H)

print("\nЗадание 1.4: Генерация кодовых слов из G:")
linear_code.print_codewords_with_input()

print("\nШаг 8: Генерация кодовых слов из всех двоичных слов:")
codewords_binary = linear_code.generate_codewords_from_binary()
print("Кодовые слова, сгенерированные из всех двоичных слов:", codewords_binary)

print("\nШаг 9: Вычисление минимального расстояния кода:")
min_distance, t = linear_code.calculate_min_distance()
print("Минимальное расстояние d =", min_distance)
print("Кратность обнаруживаемых ошибок t =", t)

