import numpy as np
import sympy as sp
import random


alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_+=[]|;:,.<>?/~ "
N = len(alphabet)
# Computationally faster to create a hash table to get the index than use alphabet.index() 
letter_to_index = {letter: index for index, letter in enumerate(alphabet)}



########################################################################################################################

def caesar_encrypt(text, key=None):
    # Generate a random key if none have been provided
    if key is None:
        key = random.randint(1,N-1)
        return key, caesar_encrypt(text,key)
    # Encrypt
    encrypted_text = ""
    for ch in text:
        encrypted_text += alphabet[(letter_to_index[ch] + key ) % N]

    return encrypted_text

def caesar_decrypt(text, key):
    decrypted_text = ""
    for ch in text:
        decrypted_text += alphabet[(letter_to_index[ch] - key ) % N]

    return decrypted_text

######################################################################################################################

def vigenere_encrypt(text, key=None):
    # Generates a random key if none have been provided
    if key is None:
        key_length = random.randint(1,len(text))
        key = ''.join([alphabet[random.randint(0,N-1)] for i in range(key_length)])
        return key, vigenere_encrypt(text, key)

    encrypted_text = ""
    for i in range(len(text)):
        encrypted_text += alphabet[(letter_to_index[text[i]] + letter_to_index[key[i % len(key)]]) % N]

    return encrypted_text

def vigenere_decrypt(text, key):
    decrypted_text = ""
    for i in range(len(text)):
        decrypted_text += alphabet[(letter_to_index[text[i]] - letter_to_index[key[i % len(key)]]) % N]

    return decrypted_text

######################################################################################################################

def affine_encrypt(text, a=None, b=None):
    # Generates a random key if none have been provided
    if a is None or b is None:
        a = 0
        while np.gcd(a,N) != 1:
            a = random.randint(2,N)
            b = random.randint(0,N-1)
        return a, b, affine_encrypt(text,a,b)

    # Checks if it will be posible to decrypt with the key a
    if np.gcd(a, N) != 1:
        return "The gcd of a={a} and " + str(N) + " (the length of the alphabet used) is not 1, hence the text will not be able to be deciphered."

    # Encrypt
    encrypted_text = ""
    for ch in text:
        encrypted_text += alphabet[(a*letter_to_index[ch] + b ) % N]
            
    return encrypted_text

def affine_decrypt(text, a, b):
    
    # Checks if it is posible to decrypt with the key a
    if np.gcd(a, N) != 1:
        return "The gcd of a={a} and " + str(N) + " (the length of the alphabet used) is not 1, hence the text will not be able to be deciphered."

    # Decrypt
    decrypted_text = ""
    inv_a = sp.mod_inverse(a, N)
    for ch in text:
        decrypted_text += alphabet[inv_a*(letter_to_index[ch] - b ) % N]

    return decrypted_text

######################################################################################################################

def hill_encrypt(text, A=None):
    # Generates a random key if none have been provided
    if A is None:
        n = np.random.randint(2,len(text))
        A = [[random.randint(0,N-1) for j in range(n)] for i in range(n)]
        while np.linalg.det(A) == 0 or np.gcd(int(np.linalg.det(A)), N) != 1:
            A = [[random.randint(0,N-1) for j in range(n)] for i in range(n)]
        return A, hill_encrypt(text, A)

    n = len(A)
    A = np.array(A)
    
    # Adds the necessary number of characters at the end of the text in order to be able to compute all operations
    if len(text) % n != 0:
        text += '~' * (n - (len(text) % n)) 

    # Encrypt
    encrypted_text = ""
    for i in range(0, len(text) - n + 1, n):
        result = (A @ ([letter_to_index[ch] for ch in text[i:i+n]])) % N
        for j in range(n):
            encrypted_text += alphabet[result[j]]
        
    return encrypted_text

def hill_decrypt(text, A):
    n = len(A)
    
    # Decrypt
    decrypted_text = ""
    invA = np.array(sp.Matrix(A).inv_mod(N))
    for i in range(0, len(text) - n + 1, n):
        result = invA @ ([letter_to_index[ch] for ch in text[i:i+n]]) % N
        for j in range(n):
            decrypted_text += alphabet[result[j]]
    
    return decrypted_text

##################################################################################################################

def railfence_encrypt(text, key=None):
    # Generates a random key if none have been provided
    if key is None:
        key = random.randint(2, len(text) // 2)
        return key, railfence_encrypt(text, key)

    # Arrange the text in a zig-zag pattern trough the rails
    T = [["."] * len(text) for i in range(key)]
    direction = -1
    i = 0
    for j in range(len(text)):
        T[i][j] = text[j]
        if i == 0 or i == key - 1:
            direction = direction * (-1)
        i += direction

    # Arrange the encrypted text by rows 
    encrypted_text = ''
    for row in T:
        encrypted_text += ''.join(char for char in row if char != '.')

    return encrypted_text

def railfence_decrypt(text, N):
    pass

####################################################################################################################

def columnar_transposition_encrypt(text, key=None):
    # Generates a random key if none have been provided
    if key is None:
        key_length = random.randint(2,len(text) // 2)
        key = ''.join([alphabet[random.randint(0,N-1)] for i in range(key_length)])
        return key, columnar_transposition_encrypt(text,key)
    # Adds character to the end of the text if it is necessary
    if len(text) % len(key) != 0:
        remaning_character = len(key) - len(text) % len(key)
        for i in range(remaning_character):
            text += alphabet[random.randint(0,N-1)]

    num_columns = len(key)
    num_rows = len(text) // len(key)
    T = [[''] * num_columns for i in range(num_rows)]
    
    # Arrange the text in matrix form
    T = [[text[i*num_columns + j] for j in range(num_columns)] for i in range(num_rows)]

    # Find the order of columns for the encryption
    key_with_index = [(char, index) for index, char in enumerate(key)]
    sorted_indices = sorted(key_with_index)

    # Concatenate the columns by order defined in sorted_index
    encrypted_text = ''
    for j in range(num_columns):
        encrypted_text += ''.join(T[i][sorted_indices[j][1]] for i in range(num_rows))
    
    return encrypted_text


def columnar_transposition_decrypt(text, key):
    
    num_columns = len(key)
    num_rows = len(text) // len(key)
    T = [[''] * num_columns for i in range(num_rows)]

    # Find the order of columns for the decryption
    key_with_index = [(char, index) for index, char in enumerate(key)]
    sorted_indices = sorted(key_with_index)

    # Arrange the text in matrix form
    l = 0
    for j in range(num_columns):
        for i in range(num_rows):
            T[i][sorted_indices[j][1]] = text[l]
            l += 1
    
    # Construct the decrypted text by concatenating the rows 
    decrypted_text = ''
    for row in T:
        decrypted_text += ''.join(char for char in row)

    return decrypted_text

####################################################################################################################

def one_time_pad_encrypt(text, key=None):
    # Generates a random key if none have been provided
    if key is None:
        key = ''.join([alphabet[random.randint(0,N-1)] for i in range(len(text))])
        return key, one_time_pad_encrypt(text, key)
    # Checks if key length is at least as text length
    if len(key) < len(text):
        return "The text cannot be encrypted, the length of the key must be equal or longer than the length of the text"
    # Encrypt
    encrypted_text = ''
    for i in range(len(text)):
        encrypted_text += alphabet[(letter_to_index[text[i]] + letter_to_index[key[i]]) % N]

    return encrypted_text

def one_time_pad_decrypt(text, key):
    # Checks if key length is at least as text length
    if len(key) < len(text):
        return "The text cannot be decrypted, the length of the key must be equal or longer than the length of the text"
    # Decrypt
    decrypted_text = ''
    for i in range(len(text)):
        decrypted_text += alphabet[(letter_to_index[text[i]] - letter_to_index[key[i]]) % N]
    return decrypted_text

#####################################################################################################################

# AES CIPHER
S_BOX = [
    [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76],
    [0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0],
    [0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15],
    [0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75],
    [0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84],
    [0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF],
    [0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8],
    [0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2],
    [0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73],
    [0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB],
    [0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79],
    [0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08],
    [0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A],
    [0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E],
    [0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF],
    [0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]
]
INV_S_BOX = [
    [0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb],
    [0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb],
    [0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e],
    [0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25],
    [0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92],
    [0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84],
    [0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06],
    [0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b],
    [0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73],
    [0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e],
    [0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b],
    [0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4],
    [0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f],
    [0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef],
    [0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61],
    [0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]
]
def aes_add_round_key(text_block, key):
    for i in range(16):
        text_block[i] = text_block[i] ^ key[i]      

def aes_sub_bytes(text_block):
    indices = text_block.hex()
    for k in range(len(text_block)):
        i = int(indices[2*k],16)
        j = int(indices[2*k+1],16)
        text_block[k] = S_BOX[i][j]

def aes_inv_sub_bytes(text_block):
    indices = text_block.hex()
    for k in range(len(text_block)):
        i = int(indices[2*k],16)
        j = int(indices[2*k+1],16)
        text_block[k] = INV_S_BOX[i][j]

def aes_shift_rows(text_block):
    temp = bytearray(text_block)
    for i in range(4):
        text_block[4*i + 1] = temp[(4*i + 1 + 4) % 16]
        text_block[4*i + 2] = temp[(4*i + 2 + 8) % 16]
        text_block[4*i + 3] = temp[(4*i + 3 + 12) % 16]
    
def aes_inv_shift_rows(text_block):
    temp = bytearray(text_block)
    for i in range(4):
        text_block[4*i + 3] = temp[(4*i + 3 + 4) % 16]
        text_block[4*i + 2] = temp[(4*i + 2 + 8) % 16]
        text_block[4*i + 1] = temp[(4*i + 1 + 12) % 16]

def aes_mix_columns(text_block):
    mix_columns_matrix = bytearray([0x2,0x3,0x1,0x1,0x1,0x2,0x3,0x1,0x1,0x1,0x2,0x3,0x3,0x1,0x1,0x2])

    # Matrices multiplication but with XOR and GF(256) multiplication as operations
    temp = 0
    result = bytearray()
    for i in range(4):
        for k in range(4):
            for j in range(4):
                temp = temp ^ GF256_multiplication(mix_columns_matrix[4*k + j], text_block[4*i + j])
            result.append(temp)
            temp = 0
    
    text_block[:] = result

def aes_inv_mix_columns(text_block):
    inv_mix_columns_matrix = [ 0x0e, 0x0b, 0x0d, 0x09,0x09, 0x0e, 0x0b, 0x0d,0x0d, 0x09, 0x0e, 0x0b,0x0b, 0x0d, 0x09, 0x0e]

    # Matrices multiplication but with XOR as sum and multiplication in GF(256)
    temp = 0
    result = []
    for i in range(4):
        for k in range(4):
            for j in range(4):
                temp = temp ^ (GF256_multiplication(inv_mix_columns_matrix[4*k + j], text_block[4*i + j]))
            result.append(temp)
            temp = 0

    text_block[:] = result

def aes_key_schedule(key):
    round_keys = []
    round_keys.append(key)
    rounds = 10
    rc = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
    key_i = bytearray(key)
    for i in range(rounds):
        w3 = bytearray([key_i[13], key_i[14], key_i[15], key_i[12]])
        aes_sub_bytes(w3)
        w3[0] = w3[0]^rc[i]
        for j in range(4):
            key_i[j] = key_i[j] ^ w3[j]
            key_i[4 + j] = key_i[4 + j] ^ key_i[j]
            key_i[8 + j] = key_i[8 + j] ^ key_i[4 + j]
            key_i[12 + j] = key_i[12 + j] ^ key_i[8 + j]
        round_keys.append(key_i[:])
        
    return round_keys

# Direct and naive approach, but it works
def GF256_multiplication(num1, num2):
    x = sp.Symbol('x')
    irreducible_polynomial = sp.Poly(x**8 + x**4 + x**3 + x + 1)
    # Transform the numbers to binary and construct the polynomial
    num1_coeff = [int(ch) for ch in bin(num1)[2:].zfill(8)]
    num2_coeff = [int(ch) for ch in bin(num2)[2:].zfill(8)]
    
    p1 = sp.Poly.from_list(num1_coeff,gens=x)
    p2 = sp.Poly.from_list(num2_coeff,gens=x)

    # Multiplication of the two polynomials and apply mod 2 to the coefficients of the result
    result_coeffs = (p1*p2).all_coeffs()
    result_coeffs = [coeff % 2 for coeff in result_coeffs]
    result = sp.Poly.from_list(result_coeffs,gens=x)

    # Compute the mod of the result by the irrecductible polynomial in GF(256)
    result = result % irreducible_polynomial

    # Transform the polynomial to binary representation
    result_coeffs = [(coeff % 2) for coeff in result.all_coeffs()]
    result = ''.join([str(num) for num in result_coeffs])
    
    return int(result,2)


def aes_encrypt(text, key=None, hex_key=False):
    
    if key is None:
        if hex_key == True:
            key = ''.join(format(random.randint(0,255),'02x') for i in range(16))
            return key, aes_encrypt(text, key, hex_key=True)
        else:
            key = ''.join(chr(random.randint(33,126)) for i in range(16))
            return key, aes_encrypt(text, key, hex_key=False)

    else:
        if hex_key == True:
            key = bytearray.fromhex(key)
            # Checks and modifies the key and length if necessary (key must have a length of 16 bytes)
            if len(key) < 16:
                # Add random bytes to the key in order to be of size 16 bytes
                key += bytearray([random.randint(0,255) for i in range(16 - len(key) % 16)]) 
                key = bytes(key).hex()
                return key, aes_encrypt(text,key,hex_key=True)

            elif len(key) > 16:
                # Remove remaining bytes in order to be of size 16 bytes
                key = bytearray(key[:16])
                key = bytes(key).hex()
                return key, aes_encrypt(text, key, hex_key=True)
        else:
            key = bytearray(key, encoding='ascii')
            # Checks and modifies the key and length if necessary (key must have a length of 16 bytes)
            if len(key) < 16:
                # Add random characters to the key in order to be of size 16 bytes
                key += bytearray([random.randint(33,126) for i in range(16 - len(key) % 16)]) # [33,126] are the range of human readable ASCII characters
                key = key.decode('ascii')
                return key, aes_encrypt(text, key, hex_key=False)

            elif len(key) > 16:
                # Remove remaining bytes in order to be of size 16 bytes
                key = bytearray(key[:16])
                key = key.decode('ascii')
                return key, aes_encrypt(text,key,hex_key=False)
            
    # Adds NULL characters to the end of the text in order to be able to divide all the text in blocks of 16 bytes
    if len(text) % 16 != 0:
        text += chr(0x00) * (16 - len(text) % 16)
    text = bytearray(text,encoding='ascii')

    # Encrypt
    encrypted_text = ''
    keys = aes_key_schedule(key)
    for i in range(len(text) // 16):
        text_block = text[i*16 : i*16 + 16]
        aes_add_round_key(text_block,keys[0])
        
        # Rounds encryption
        for i in range(1,10):
            aes_sub_bytes(text_block)
            aes_shift_rows(text_block)
            aes_mix_columns(text_block)
            aes_add_round_key(text_block,keys[i])
            
        # Last round
        aes_sub_bytes(text_block)
        aes_shift_rows(text_block)
        aes_add_round_key(text_block,keys[-1])

        encrypted_text += bytes(text_block).hex()

    return encrypted_text

def aes_decrypt(text, key, hex_key=False):

    if hex_key == True:
        key = bytearray.fromhex(key)
    else:
        key = bytearray(key,encoding='ascii')
    if len(key) != 16:
        return "The length of the key is not 16, unable to compute the decryption"

    text = bytearray.fromhex(text)

    # Decrypt
    decrypted_text = ''
    keys = aes_key_schedule(key)
    for i in range(len(text) // 16):
        text_block = text[16*i:16*i+16]

        # First round
        aes_add_round_key(text_block, keys[-1])
        aes_inv_shift_rows(text_block)
        aes_inv_sub_bytes(text_block)

        # Rounds encryption
        for i in range(1,10):
            aes_add_round_key(text_block, keys[-(i+1)])
            aes_inv_mix_columns(text_block)
            aes_inv_shift_rows(text_block)
            aes_inv_sub_bytes(text_block)

        aes_add_round_key(text_block, keys[0])

        # Transform the numbers in characters ignoring the NULLS added in the encryption process
        decrypted_text += ''.join(chr(num) for num in text_block if num != 0x00)
    
    return decrypted_text

################################################################################################################

DES_S_BOXES = [
    # S1
    [   [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    # S2
    [   [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    # S3
    [   [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    # S4
    [   [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    # S5
    [   [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    # S6
    [   [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    # S7
    [   [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    # S8
    [   [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]

DES_IP = [
    58,	50,	42,	34,	26,	18,	10,	2,
    60,	52,	44,	36,	28,	20,	12,	4,
    62,	54,	46,	38,	30,	22,	14,	6,
    64,	56,	48,	40,	32,	24,	16,	8,
    57,	49,	41,	33,	25,	17,	9,	1,
    59,	51,	43,	35,	27,	19,	11,	3,
    61,	53,	45,	37,	29,	21,	13,	5,
    63,	55,	47,	39,	31,	23,	15,	7
]
DES_IP_INV = [
    40,	8,	48,	16,	56,	24,	64,	32,
    39,	7,	47,	15,	55,	23,	63,	31,
    38,	6,	46,	14,	54,	22,	62,	30,
    37,	5,	45,	13,	53,	21,	61,	29,
    36,	4,	44,	12,	52,	20,	60,	28,
    35,	3,	43,	11,	51,	19,	59,	27,
    34,	2,	42,	10,	50,	18,	58,	26,
    33,	1,	41,	9,	49,	17,	57,	25
]
DES_PC1 = [
    57, 49, 41, 33, 25, 17,  9,
    1,  58, 50, 42, 34, 26, 18,
    10, 2,  59, 51, 43, 35, 27,
    19, 11, 3,  60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7,  62, 54, 46, 38, 30, 22,
    14, 6,  61, 53, 45, 37, 29,
    21, 13, 5,  28, 20, 12, 4
]
DES_PC2 = [
    14, 17, 11, 24, 1,  5,
    3,  28, 15, 6,  21, 10,
    23, 19, 12, 4,  26, 8,
    16, 7,  27, 20, 13, 2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
]
DES_E = [
    32, 1,  2,  3,  4,  5,
    4,  5,  6,  7,  8,  9,
    8,  9,  10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]
DES_P = [
    16, 7, 20, 21,
    29, 12, 28, 17,
    1, 15, 23, 26,
    5, 18, 31, 10,
    2, 8, 24, 14,
    32, 27, 3, 9,
    19, 13, 30, 6,
    22, 11, 4, 25
]

def des_initial_permutation(text):
    temp = text[:]
    # Computes permutation with IP table
    for i in range(64):
        text[i] = temp[DES_IP[i]-1]

def des_final_permutation(text):
    temp = text[:]
    # Computes permutation with IP^{-1} table
    for i in range(64):
        text[i] = temp[DES_IP_INV[i]-1]
        
def des_key_schedule(key):
    # Computes permutation of the key with PC1 table
    key_permuted = [key[DES_PC1[i]-1] for i in range(56)]

    # Splits the key in two halves
    C_0 = key_permuted[:28]
    D_0 = key_permuted[28:]
    # Vector that stores the number of left shits apply to each halve in each iteration
    number_of_shifts = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]
    # Lists to store the halves of all iterations
    left_halves = []
    right_halves = []
    left_halves.append(C_0)
    right_halves.append(D_0)

    # In each iteration each half is the previous half but circular shifted every bit left
    for i in range(16):
        C_i = left_halves[i][number_of_shifts[i]:] + left_halves[i][:number_of_shifts[i]]
        D_i = right_halves[i][number_of_shifts[i]:] + right_halves[i][:number_of_shifts[i]]
        left_halves.append(C_i)
        right_halves.append(D_i)
    round_keys = []
    for i in range(1,17):
        temp = (left_halves[i] + right_halves[i])[:]
        # Computes permutation of key_i with PC2 table
        key_i = [temp[DES_PC2[j]-1] for j in range(48)]
        round_keys.append(key_i)

    return round_keys
    
def des_feistel_round(text, round_key):
    # L_n = R_{n-1}
    L_n = text[32:]
    f = []
    # Computes permutation of R_{n-1} with the E_bit table (Notice that R_{n-1} = L_n)
    for i in range(48):
        f.append(L_n[DES_E[i]-1]^round_key[i])
    # For each block B_i of 6 bytes from f, compute S_i_Box substitution (S(B_k) = S_k_Box[i][j], where 
    # i := concatenate the  first and last byte of B_i and interprete it as decimal number
    # j := concatenate the 4 middle bytes of B_i and interprete them as decimal number
    temp_f = []
    for k in range(8):
        B = f[6*k:6*k + 6]
        i = int(''.join(str(num) for num in [B[0],B[-1]]), 2)
        j = int(''.join(str(num) for num in B[1:5]), 2)
        SB = DES_S_BOXES[k][i][j]
        temp_f += [int(ch) for ch in bin(SB)[2:].zfill(4)]
    
    # Apply permutation of f with the P table
    f.clear()
    for i in range(32):
        f.append(temp_f[DES_P[i]-1])

    # Computes R_{n} = L_{n-1} XOR f
    R_n = [text[i]^f[i] for i in range(32)]

    # Concatenates L_n and R_n
    text = L_n + R_n

    return text
    
def des_encrypt(text, key):
    # Transform string of ASCII characters to hexadecimal string of length 16 (64 bits)
    text = ''.join([format(ord(ch),'02x') for ch in text])
    key = ''.join([format(ord(ch),'02x') for ch in key])
    
    if len(key) < 16:
        print("Key length is less than 64 bits, random characters have been added to the key")
        key += ''.join([format(random.randint(41, 126),'02x') for i in range(16 - len(key) % 16)])
        print("New Key:", ''.join([chr(int(key[i:i+2],16)) for i in range(0,len(key)-1,2)]))
    if len(key) > 16:
        print("Key length is greater than 64 bits, remaining characters have been removed")
        key = key[:16]
        print("New Key:", ''.join([chr(int(key[i:i+2],16)) for i in range(0,len(key)-1,2)] ) )

    if len(text) % 16 != 0:
        text += format(0,'02x') * (16 - len(text) % 16)
        
        
    # Transform string of hexadecimals to vector of bits
    encrypted_text = ""
    # Transform string of hexadecimals to list of binary digits
    key = [int(bit) for char in key for bit in f"{int(char, 16):04b}"]
    round_keys = des_key_schedule(key)

    for i in range(len(text) // 16):
        # Transform string of hexadecimals to list of binary digits
        block_text = [int(bit) for char in text[16*i:16*i+16] for bit in f"{int(char, 16):04b}"]
        
        des_initial_permutation(block_text)
        # Applies 16 rounds of feistel
        for i in range(16):
            block_text = des_feistel_round(block_text,round_keys[i])

        # Swaps the two halves for the final permutation
        left_half = block_text[32:]
        right_half = block_text[:32]
        block_text[:] = left_half + right_half
        des_final_permutation(block_text)

        # Transforms the 64 bits text encryted in a string of hexadecimals
        for k in range(8):
            e = ''.join(map(str,block_text[8*k:8*k+8]))
            encrypted_text += format(int(e,2),'02x')

    return encrypted_text

def des_decrypt(text, key):
    
    # Transform the ascii key into string of hexadecimals and transform it to list of binary digits 
    key = ''.join([format(ord(ch),'02x') for ch in key])

    # Transform string of hexadecimals to list of binary digits
    key = [int(bit) for char in key for bit in f"{int(char, 16):04b}"]
    decrypted_text = ""
    # Computes all the round keys 
    round_keys = des_key_schedule(key)
    for i in range(len(text) // 16):
        # Transform string of hexadecimals to list of binary digits
        block_text = [int(bit) for char in text[16*i:16*i+16] for bit in f"{int(char, 16):04b}"]
        # Decryption
        des_initial_permutation(block_text)
        # Applies 16 rounds of feistel 
        for i in range(16):
            block_text = des_feistel_round(block_text, round_keys[-(i+1)])
        # Swaps the two halves for the final permutation
        left_half = block_text[32:]
        right_half = block_text[:32]
        block_text[:] = left_half + right_half
        des_final_permutation(block_text)

        # Transforms the 64 bits encrypted text into a string of ASCII characters
        for k in range(8):
            e = ''.join(map(str, block_text[8*k:8*k+8]))
            # Ignores the NULL characters added in the encryption process
            if int(e,2) != 0x00:
                decrypted_text += chr(int(e,2))

    return decrypted_text

#################################################################################################################

def triple_des_encrypt(text, key1, key2, key3):
    return des_encrypt(des_decrypt(des_encrypt(text,key1),key2),key3)


def triple_des_decrypt(text, key1, key2, key3):
    return des_decrypt(des_encrypt(des_decrypt(text,key3),key2),key1)

#############################################################################################################

def RC4_encrypt(text, key=None, hex_key=False):

    if key is None:
        if hex_key == True:
            n = random.randint(1,256)
            key = ''.join([format(random.randint(0,256),'02x') for i in range(n)])
            return key, RC4_encrypt(text, key, hex_key=True)

    if hex_key == True:
        key = bytearray.fromhex(key)
    else:
        key = bytearray(key, encoding='ascii')

    # We'll use RC4_encrypt for decryption too, so when decrypting, type(text) is going to be a bytearray already
    if type(text) == str:
        text = bytearray(text, encoding='ascii')

    S = bytearray(range(256))
    key_length =  len(key)
    # Key schedule
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % key_length]) % 256
        S[i] , S[j] = S[j], S[i]

    # Pseudo-random generation algorithm
    i = 0
    j = 0
    for i in range(len(text)):  
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i] , S[j] = S[j], S[i]
        K = S[(S[i] + S[j]) % 256]
        # XOR the byte K with text
        text[i-1] = text[i-1] ^ K
    # Transform the bytearray into a string of hexadecimals
    #encrypted_text = ''.join(format(byte, '02x') for byte in text)
    encrypted_text = bytes(text).hex()

    return encrypted_text

def RC4_decrypt(text, key, hex_key=False):
    # Encryption is the same as decryption in RC4
    text = bytearray.fromhex(text)
    decrypted_text = RC4_encrypt(text, key, hex_key)
    # Transform the string of hexadecimals to a string of ascii characters
    decrypted_text = bytes.fromhex(decrypted_text).decode('ascii')

    return decrypted_text

######################################################################################################################################################


    
#########################################################################################################################################################
def main():
    # Tests

    #print('This is a test'* 43 == caesar_decrypt(caesar_encrypt('This is a test'* 43,53),53))
    #print('This is a test' * 43 ==  vigenere_decrypt(vigenere_encrypt('This is a test' * 43,'keytest!'),'keytest!'))
    #print('This is a test' * 43 == affine_decrypt(affine_encrypt('This is a test' * 43,3,7),3,7))
    #print(hill_decrypt(hill_encrypt("Prueba de hill Decipher ol12321!.,", [[1, 2, 3],[4, 5, 6],[7, 8, 20]]),[[1, 2, 3],[4, 5, 6],[7, 8, 20]]))
    #print(railfence_decrypt(railfence_encrypt("We are fucking discovered!, hell yeah!!",4),4))
    #print(columnar_transposition_decrypt(columnar_transposition_encrypt('This is a test' * 2,"ZEBRAS"),"ZEBRAS"))
    #print('This is a test' * 43 == one_time_pad_decrypt(one_time_pad_encrypt('This is a test' * 43,'This is a test' * 43),'This is a test' * 43))

    #print("Two One Nine Two" * 10 == aes_decrypt(aes_encrypt("Two One Nine Two" * 10,"Thats my Kung Fu"), "Thats my Kung Fu"))
    #print('ahjhyuytrdd6888ffasdasASDASW325' == des_decrypt((des_encrypt('ahjhyuytrdd6888ffasdasASDASW325','abcdefgh')),'abcdefgh'))
    #print('This is a test of 3DES, does it work?' ==  triple_des_decrypt(triple_des_encrypt('This is a test of 3DES, does it work?','abcdefgh','abcdefgh','abcdefgh'),'abcdefgh','abcdefgh','abcdefgh'))
    #print('The quick brown fox jumps over the lazy dog.' == RC4_decrypt(RC4_encrypt('The quick brown fox jumps over the lazy dog.','cryptii'),'cryptii'))
    #print(RC4_encrypt('This is a fucking test!','criptokey'))

    # 32 byte key
    print(des_encrypt('This is a test','this '))



    
if __name__ == "__main__":
    main()