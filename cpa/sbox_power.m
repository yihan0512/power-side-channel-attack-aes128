function power_pattern = sbox_power(text, key, sbox)

xor_result = (bitxor(text, key));
sbox_result = sbox(xor_result+1);
power_pattern = sum((dec2bin(sbox_result) == '1'), 2);