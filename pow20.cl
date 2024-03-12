typedef uint u;
typedef ulong ul;

__constant u K[64] = { 
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant u H[8] = { 
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19
};

#define rot(x, y) rotate(x, (u)y)
#define ch(x, y, z) bitselect(z, y, x)
#define s0(x) (rot(x, 25) ^ rot(x, 14) ^ (x >> 3))
#define s1(x) (rot(x, 15) ^ rot(x, 13) ^ (x >> 10))
#define S1(x) (rot(x, 26) ^ rot(x, 21) ^ rot(x, 7))
#define S0(x) (rot(x, 30) ^ rot(x, 19) ^ rot(x, 10))
#define maj(x, y, z) ch((z ^ x), y, x)
#define rotr(x, y) rotate((u)x, (u)(32-y))

void sha256_transform(const u* data, u* hash) {
    u W[64];
    u a, b, c, d, e, f, g, h, temp1, temp2;

    a = hash[0];
    b = hash[1];
    c = hash[2];
    d = hash[3];
    e = hash[4];
    f = hash[5];
    g = hash[6];
    h = hash[7];

    for (int i = 0; i < 16; ++i) {
        W[i] = data[i];
    }

    for (int i = 16; i < 64; i++) {
        W[i] = W[i - 16] + s0(W[i - 15]) + W[i - 7] + s1(W[i - 2]);
    }

    for (uint i = 0; i < 64; i++) {
        temp1 = h + S1(e) + ch(e, f, g) + K[i] + W[i];
        temp2 = S0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
}

__kernel void pow20(__global ul* result, __global u* input, ul start_nonce) {
    int index = get_global_id(0);
    ul nonce = start_nonce + index;
    u data[16];

    data[0] = input[7];
    data[1] = input[6];
    data[2] = input[5];
    data[3] = input[4];
    data[4] = input[3];
    data[5] = input[2];
    data[6] = input[1];
    data[7] = input[0];
    data[8] = (uint)((nonce >> 32) & 0xFFFFFFFF);
    data[9] = (uint)(nonce & 0xFFFFFFFF);
    data[10] = 0x80000000,
    data[11] = 0x00000000;
    data[12] = 0x00000000;
    data[13] = 0x00000000;
    data[14] = 0x00000000;
    data[15] = 0x00000140;

    u vals[8];
    vals[0] = H[0];
    vals[1] = H[1];
    vals[2] = H[2];
    vals[3] = H[3];
    vals[4] = H[4];
    vals[5] = H[5];
    vals[6] = H[6];
    vals[7] = H[7];
    sha256_transform(data, vals);
    
    data[0] = vals[0];
    data[1] = vals[1];
    data[2] = vals[2];
    data[3] = vals[3];
    data[4] = vals[4];
    data[5] = vals[5];
    data[6] = vals[6];
    data[7] = vals[7];
    data[8] = 0x80000000,
    data[9] = 0x00000000;
    data[10] = 0x00000000;
    data[11] = 0x00000000;
    data[12] = 0x00000000;
    data[13] = 0x00000000;
    data[14] = 0x00000000;
    data[15] = 0x00000100;

    vals[0] = H[0];
    vals[1] = H[1];
    vals[2] = H[2];
    vals[3] = H[3];
    vals[4] = H[4];
    vals[5] = H[5];
    vals[6] = H[6];
    vals[7] = H[7];

    sha256_transform(data, vals);

    if (vals[0] == 0) {
        result[++result[0]] = nonce;
    }
}
