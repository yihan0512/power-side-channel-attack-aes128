% loadData
num_trace = 500;
attack_section = 5000;
trace_pattern = trace(1:num_trace, 1:attack_section);
trace_pattern = trace_pattern - repmat(mean(trace_pattern), [num_trace, 1]);
trace_pattern = trace_pattern ./ repmat(std(trace_pattern), [num_trace, 1]);
key = zeros(16, 1);

for subkey = 1:16
    disp(['processing #', num2str(subkey), ' subkey...'])
    corr_result = zeros(256, size(trace_pattern, 2));
    text_input = textin(1:num_trace, subkey);
    parfor subkey_guess = 0:255
        xor_result = (bitxor(text_input, subkey_guess));
        sbox_result = sbox(xor_result+1);
        power_pattern = sum((dec2bin(sbox_result) == '1'), 2);
        power_pattern = power_pattern - mean(power_pattern);
        power_pattern = power_pattern / std(power_pattern);
        corr_result(subkey_guess+1, :) = power_pattern' * trace_pattern;
    end
    [~, key(subkey)] = max(max(abs(corr_result), [], 2));
end
key = key - 1;
%% validation
if 0
    for i = 1:10
        disp(['testing #' num2str(i) ' plaintext...'])
        ciphertext(i, :) = aes_demo(key, textin(i, :));
    end
    expected_ciphertext = text(1:10, 17:end);

    ciphertext == expected_ciphertext
end
%% validation on test data
if 0
    validation_range = 981:990;
    ciphertext = zeros(length(validation_range), 16);
    flag = 1;
    for i = validation_range
        disp(['testing #' num2str(i) ' plaintext...'])
        ciphertext(flag, :) = aes_demo(key, text_test(i, 1:16));
        flag = flag +1;
    end
    expected_ciphertext = text_test(validation_range, 17:end);

    ciphertext == expected_ciphertext
end