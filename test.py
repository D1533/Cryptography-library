import criptography_library as cr
import random
import numpy as np

def main():
   
    # Generates random texts and keys to test all ciphers
    for i in range(25):

        plain_text = ''.join([chr(random.randint(32,126)) for i in range(32)])
        caesar_key = random.randint(1,cr.N)
        vigenere_key = ''.join([cr.alphabet[random.randint(0, cr.N-1)] for i in range(random.randint(1,len(plain_text)))])

        affine_key = [random.randint(2,cr.N), random.randint(0,cr.N-1)]
        while np.gcd(affine_key[0],cr.N) != 1:
            affine_key[0] = random.randint(2,cr.N)

        columnar_transposition_key = ''.join([cr.alphabet[random.randint(0, cr.N-1)] for i in range(len(plain_text)//2)])
        one_time_pad_key = ''.join([cr.alphabet[random.randint(0,cr.N-1)] for i in range(len(plain_text))])
        AES_key = ''.join(format(random.randint(0,255),'02x') for i in range(16))
        DES_key = ''.join(format(random.randint(0,255),'02x') for i in range(16))
        RC4_key = ''.join([format(random.randint(0,256),'02x') for i in range(random.randint(1,256))])

        if plain_text != cr.caesar_decrypt(cr.caesar_encrypt(plain_text, caesar_key),caesar_key):
            print('Caesar cipher Error')
        if plain_text != cr.vigenere_decrypt(cr.vigenere_encrypt(plain_text, vigenere_key),vigenere_key):
            print('Vigenere cipher Error')
        if plain_text != cr.affine_decrypt(cr.affine_encrypt(plain_text, affine_key[0], affine_key[1]),affine_key[0], affine_key[1]):
            print('Affine cipher Error')
        if plain_text != cr.columnar_transposition_decrypt(cr.columnar_transposition_encrypt(plain_text, columnar_transposition_key), columnar_transposition_key):
            print('Columnar transposition cipher Error')
        if plain_text != cr.one_time_pad_decrypt(cr.one_time_pad_encrypt(plain_text, one_time_pad_key), one_time_pad_key):
            print('One time pad cipher Error')
        if plain_text != cr.aes_decrypt(cr.aes_encrypt(plain_text, AES_key, True), AES_key, True):
            print('AES cipher Error')
        if plain_text != cr.des_decrypt(cr.des_encrypt(plain_text, DES_key), DES_key):
            print('DES cipher Error')
        if plain_text != cr.triple_des_decrypt(cr.triple_des_encrypt(plain_text, DES_key,DES_key,DES_key), DES_key,DES_key,DES_key):
            print('3DES cipher Error')
        if plain_text != cr.RC4_decrypt(cr.RC4_encrypt(plain_text, RC4_key), RC4_key):
            print('RC4 cipher Error')




    
if __name__ == "__main__":
    main()