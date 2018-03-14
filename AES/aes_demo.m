function ciphertext = aes_demo(key, plaintext)

%AES_DEMO  Demonstration of AES-components.
%
%   AES_DEMO
%   runs a demonstration of all components of 
%   the Advanced Encryption Standard (AES) toolbox.
%
%   In the initialization step the S-boxes, the round constants,
%   and the polynomial matrices are created and
%   an example cipher key is expanded into 
%   the round key schedule.
%   Step two and three finally convert 
%   an example plaintext to ciphertext and back to plaintext.

%   Copyright 2001-2005, J. J. Buchholz, Hochschule Bremen, buchholz@hs-bremen.de

%   Version 1.0     30.05.2001

% Initialization
[s_box, inv_s_box, w, poly_mat, inv_poly_mat] = aes_init(key);

% Define an arbitrary series of 16 plaintext bytes 
% in hexadecimal (string) representation
% The following two specific plaintexts are used as examples 
% in the AES-Specification (draft)
% plaintext_hex = {'00' '11' '22' '33' '44' '55' '66' '77' ...
%                  '88' '99' 'aa' 'bb' 'cc' 'dd' 'ee' 'ff'};
%plaintext_hex = {'32' '43' 'f6' 'a8' '88' '5a' '30' '8d' ...
%                 '31' '31' '98' 'a2' 'e0' '37' '07' '34'};

% Convert plaintext from hexadecimal (string) to decimal representation
% plaintext = hex2dec (plaintext_hex);

% This is the real McCoy.
% Convert the plaintext to ciphertext,
% using the expanded key, the S-box, and the polynomial transformation matrix
ciphertext = cipher (plaintext, w, s_box, poly_mat);

% Convert the ciphertext back to plaintext
% using the expanded key, the inverse S-box, 
% and the inverse polynomial transformation matrix
% re_plaintext = inv_cipher (ciphertext, w, inv_s_box, inv_poly_mat, 1);