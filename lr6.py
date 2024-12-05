import numpy as np
import random


def encode_message(message, generator):
    return np.polymul(message, generator) % 2


def introduce_errors(message, error_count):
    n = len(message)
    error_positions = random.sample(range(n), error_count)
    print(f"Позиции ошибок: {error_positions}")
    for pos in error_positions:
        message[pos] ^= 1
    return message


def introduce_pack_errors(message, packet_length):
    n = len(message)
    start_pos = random.randint(0, n - packet_length)
    for i in range(packet_length):
        message[(start_pos + i) % n] ^= 1
    print(f"Пакет ошибок внесён в позиции от {start_pos} до {(start_pos + packet_length - 1) % n}")
    return message


def can_correct_error(error, max_length):
    error = np.trim_zeros(error, 'f')
    error = np.trim_zeros(error, 'b')
    return 0 < len(error) <= max_length


def decode_message(received, generator, max_errors, is_packet):
    n = len(received)
    syndrome = np.polydiv(received, generator)[1] % 2

    for i in range(n):
        error_vector = np.zeros(n, dtype=int)
        error_vector[n - i - 1] = 1
        multiplied_syndrome = np.polymul(syndrome, error_vector) % 2

        syndrome_i = np.polydiv(multiplied_syndrome, generator)[1] % 2

        if is_packet:
            if can_correct_error(syndrome_i, max_errors):
                return correct_message(received, syndrome_i, i, generator)
        else:
            if sum(syndrome_i) <= max_errors:
                return correct_message(received, syndrome_i, i, generator)

    return None


def correct_message(received, syndrome_i, i, generator):
    n = len(received)
    error_position = np.zeros(n, dtype=int)
    error_position[i - 1] = 1
    correction = np.polymul(error_position, syndrome_i) % 2
    corrected = np.polyadd(correction, received) % 2
    return np.array(np.polydiv(corrected, generator)[0] % 2).astype(int)


def investigate_code_7_4():
    print("-------------------------------\nИсследование кода (7,4)\n")
    generator = np.array([1, 1, 0, 1])
    max_errors = 1

    for error_count in range(1, 4):
        word = np.array([1, 0, 1, 0])
        print(f"Исходное сообщение: {word}")
        codeword = encode_message(word, generator)
        print(f"Закодированное сообщение: {codeword}")
        codeword_with_errors = introduce_errors(codeword.copy(), error_count)
        print(f"Сообщение с ошибками: {codeword_with_errors}")
        decoded = decode_message(codeword_with_errors, generator, max_errors, is_packet=False)
        print(f"Декодированное сообщение: {decoded}")
        if np.array_equal(word, decoded):
            print("Исходное сообщение и декодированное совпадают.\n")
        else:
            print("Исходное сообщение и декодированное не совпадают.\n")


def investigate_code_15_9():
    print("-------------------------------\nИсследование кода (15,9)\n")
    generator = np.array([1, 0, 0, 1, 1, 1, 1])
    max_errors = 3

    for packet_length in range(1, 5):
        word = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0])
        print(f"Исходное сообщение: {word}")
        codeword = encode_message(word, generator)
        print(f"Закодированное сообщение: {codeword}")
        codeword_with_pack_errors = introduce_pack_errors(codeword.copy(), packet_length)
        print(f"Сообщение с пакетом ошибок: {codeword_with_pack_errors}")
        decoded = decode_message(codeword_with_pack_errors, generator, max_errors, is_packet=True)
        print(f"Декодированное сообщение: {decoded}")
        if np.array_equal(word, decoded):
            print("Исходное сообщение и декодированное совпадают.\n")
        else:
            print("Исходное сообщение и декодированное не совпадают.\n")


if __name__ == '__main__':
    investigate_code_7_4()
    investigate_code_15_9()
