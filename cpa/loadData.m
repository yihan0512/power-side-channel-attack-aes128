% please specific the data path to run the code
datapath = fullfile('/', 'freespace', 'local', 'yh482')
%% read text
disp('reading text..')
text_dir = fullfile(datapath, 'AES128_ciphertext_plaintext.txt');
textfile = fopen(text_dir);
text = fscanf(textfile, '%x');
fclose(textfile);
text = reshape(text, [32 9000])';
textin = text(:, 1:16);
%% read trace
disp('reading trace..')
trace_dir = fullfile(datapath, 'AES128.txt');
trace = importdata(trace_dir, ',');
trace = trace(:, 2:end)';
%% read sbox
disp('reading sbox..')
sboxfile = fopen('sbox.txt');
sbox = fscanf(sboxfile, '%x');
fclose(sboxfile);
sbox = reshape(sbox, [16 16]);
%% read test text
disp('reading test text..')
text_dir = fullfile(datapath, 'AES128_ciphertext_plaintext_Test.txt');
textfile = fopen(text_dir);
text_test = fscanf(textfile, '%x');
fclose(textfile);
text_test = reshape(text_test, [32 1000])';